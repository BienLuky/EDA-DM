import argparse, os, glob, datetime, yaml, sys
sys.path.append('./')
print(sys.path)
import logging
import torch
import torch.nn as nn
import time
import random
import numpy as np
from tqdm import trange
# from pytorch_lightning import seed_everything
from qdiff.utils import seed_everything
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config

from qdiff import QuantModel, set_act_quantize_params_LDM, set_weight_quantize_params_LDM, recon_block_Qmodel, Change_LDM_model_attnblock
from qdiff.utils import AttentionMap

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

from task_config import bedroom_get_parser
from calibration import TDAC_bedroom_calib_data_generator

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample

def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                logger.info(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs

@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )

@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0):
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False):
    log = dict()
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    # with model.ema_scope("Plotting"):
    with torch.no_grad():
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                            make_prog_row=True)
        elif dpm:
            logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
            sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)
        t1 = time.time()
        x_sample = model.decode_first_stage(sample)
    torch.cuda.empty_cache()
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    # logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = 0
    if model.cond_stage_model is None:
        all_images = []

        logger.info(f"Running unconditional sampling for {n_samples} samples")
        with torch.no_grad():
            for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=custom_steps,
                                                eta=eta, dpm=dpm)
                n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
                torch.cuda.empty_cache()
    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = bedroom_get_parser()
    args, unknown = parser.parse_known_args()

    # fix random seed
    seed_everything(args.seed)
    torch.cuda.set_device(args.device)
    logger.info(torch.cuda.current_device())

    if not os.path.exists(args.resume):
        raise ValueError("Cannot find {}".format(args.resume))
    if os.path.isfile(args.resume):
        try:
            logdir = '/'.join(args.resume.split('/')[:-1])
            logger.info(f'Logdir is {logdir}')
        except ValueError:
            paths = args.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = args.resume
    else:
        assert os.path.isdir(args.resume), f"{args.resume} is not a directory"
        logdir = args.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    args.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    if gpu:
        device = "cuda"
    eval_mode = True

    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir)
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir
    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': not args.a_sym, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': not args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}
        qnn = QuantModel(
            model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
            sm_abit=args.sm_abit)
        qnn.cuda()
        qnn.eval()

        logger.info("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()
        qnn.disable_network_output_quantization()
        qnn.set_quant_state(False, False)
        model.model.diffusion_model = qnn

        logger.info("sampling calib data")
        cali_data = TDAC_bedroom_calib_data_generator(
            model,
            args,
            args.calib_num_samples,
            args.batch_samples,
            device,
            args.custom_steps,
        )

        if args.split:
            model.model.diffusion_model.model.split_shortcut = True

        set_weight_quantize_params_LDM(model, cali_data, args)
        set_act_quantize_params_LDM(model, cali_data, args)

        if args.recon == True:
            Change_LDM_model_attnblock(model.model.diffusion_model, aq_params)
            kwargs = dict(cali_data=cali_data[:-1], 
                            iters=5000,
                            act_quant=True, 
                            asym=True,
                            opt_mode='mse', 
                            lr_a=args.lr_a,
                            lr_w=args.lr_w,
                            p=2.0,
                            weight=0.0001,
                            b_range=(20,2), 
                            warmup=0.2,
                            batch_size=32,
                            input_prob=0.5,
                            add_loss=args.add_loss,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False
                            )
            model.model.diffusion_model.set_quant_state(True, True)
            recon_qnn = recon_block_Qmodel(args, model.model.diffusion_model, cali_data, kwargs)
            model.model.diffusion_model = recon_qnn.recon()

        model.model.diffusion_model.set_quant_state(True, True)

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(args)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)

    if args.verbose:
        logger.info(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)

    run(model, imglogdir, eta=args.eta,
        vanilla=args.vanilla_sample,  n_samples=args.n_samples, custom_steps=args.custom_steps,
        batch_size=args.batch_size, dpm=args.dpm)

    logger.info("The para add_loss is {}".format(args.add_loss))
    logger.info("The para lr_w is {}".format(args.lr_w))
    logger.info("The para lr_a is {}".format(args.lr_a))
    logger.info("done.")
