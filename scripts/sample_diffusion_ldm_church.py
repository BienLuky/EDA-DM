import argparse, os, glob, datetime, yaml, sys
sys.path.append('./pytorch-fid/src')
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

from qdiff import QuantModel, set_act_quantize_params_LDM, set_weight_quantize_params_LDM, recon_Qmodel, Change_LDM_model_attnblock
from qdiff import QuantModel
from qdiff.utils import resume_cali_model, AttentionMap
from pytorch_fid.fid_score import calculate_fid_given_paths

logger = logging.getLogger(__name__)

rescale = lambda x: (x + 1.) / 2.

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=10000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--lamda",
        type=float,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="evaluation logdir",
        default="/dataset/LSUN/churchs/train_new"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    # qdiff specific configs
    parser.add_argument(
        "--device", type=str,
        default="cuda:0",
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--dpm", action="store_true",
        help="use dpm solver for sampling"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )

    parser.add_argument(
        "--calib_im_mode",
        default="random",
        type=str,
        choices=["noise_backward_t", "greedy", "Q_diffusion"],
    )
    parser.add_argument(
        "--calib_t_mode",
        default="random",
        type=str,
        choices=['normal' , 'Qdiff'],
    )
    parser.add_argument(
        "--calib_num_samples",
        default=1024,
        type=int,
        help="size of the calibration dataset",
    )
    parser.add_argument(
        "--batch_samples",
        default=1024,
        type=int,
        help="size of the sample dataset",
    )
    parser.add_argument(
        "--class_cond", action="store_true",
        help="class difusion"
    )
    parser.add_argument(
        "--recon", action="store_true",
        help="use reconstruction"
    )
    parser.add_argument(
        "--add_loss",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma"
    )
    parser.add_argument(
        "--lr_w",
        type=float,
        default=5e-1,
        help="eta used to control the variances of sigma"
    )
    parser.add_argument(
        "--lr_a",
        type=float,
        default=1e-3,
        help="eta used to control the variances of sigma"
    )
    return parser

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
                print(f"Unknown format for key {k}. ")
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
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
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

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
    n_samples=50000, dpm=False):
    if vanilla:
        logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    # n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    n_saved = 0
    # path = logdir
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

def generate_t(t_mode, num_samples, num_timesteps, device):
    if t_mode == "normal":
        shape = torch.Tensor(num_samples)
        normal_val = torch.nn.init.normal_(shape, mean=0.4, std=0.4)*num_timesteps
        x_min, x_max = torch._aminmax(normal_val)
        normal_val = (normal_val-x_min)/(x_max-x_min)*num_timesteps
        t = normal_val.clone().type(torch.int).to(device=device)
    elif t_mode == "Qdiff":
        t = []
        vert = 5
        steps = num_timesteps/vert
        num = int(num_samples/steps)
        for time in range(num_timesteps):
            if time%vert==0 and time!=(num_timesteps-1):
                t.append(torch.full((num, ), time).int())
            elif time==(num_timesteps-1):
                t.append(torch.full((num_samples - int(num*steps), ), time).int())
        t = torch.hstack(t).to(device=device)
        idx = random.sample(range(len(t)), len(t))
        t = t[idx].to(device)
    else:
        raise NotImplementedError
    return t.clamp(0, num_timesteps - 1)

def backward_t_calib_data_generator(
    model, args, calib_num_samples, num_samples, device, t_mode, num_timesteps
):

    t = generate_t(t_mode, calib_num_samples, num_timesteps, device).long()
    PTQ_t_num = torch.zeros(num_timesteps)
    for i in t:
        PTQ_t_num[i] += 1
    assert torch.sum(PTQ_t_num) == calib_num_samples
    x = range(len(PTQ_t_num))
    f = plt.figure()
    plt.plot(x, PTQ_t_num.to('cpu'))
    plt.savefig('./Church_PTQ_t_num.png')

    # num_samples = 8
    samples = []
    ts = None
    shape = [num_samples,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]
    # sample, intermediates = convsample_ddim(model,  steps=args.custom_steps, shape=shape,eta=args.eta)
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)

            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
        torch.cuda.empty_cache()
    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()

    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index

def backward_featrue_greedy_calib_data_generator(
    model, args, calib_num_samples, num_samples, device, num_timesteps
):

    samples = []
    ts = None
    shape = [num_samples,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]
    # sample, intermediates = convsample_ddim(model,  steps=args.custom_steps, shape=shape,eta=args.eta)
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    hooks = []
    # feature_map = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                sample, intermediates, feature_map = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False, hooks=hooks,)
            else:
                sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
        torch.cuda.empty_cache()
    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    print("caculate density num")
    dense_r = 0.3
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                # print(mse)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    x = range(len(dense_num))
    f = plt.figure()
    plt.plot(x, dense_num)
    plt.savefig('./Church_dense_num.png')
    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())

    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))
    x = range(len(Cos_dis))
    f = plt.figure()
    plt.plot(x, Cos_dis.to('cpu'))
    plt.savefig('./Church_Cos_dis.png')
    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    # l = 1.0
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    x = range(len(w))
    f = plt.figure()
    plt.plot(x, w.to('cpu'))
    plt.savefig('./Church_w.png')

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] >= 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./Church_TDAC_t_num.png')

    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)
    
    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index


def Q_diffusion_calib_data_generator(
    model, args, calib_num_samples, num_samples, device, t_mode, num_timesteps
):

    t = generate_t(t_mode, calib_num_samples, num_timesteps, device).long()

    samples = []
    ts = None
    shape = [num_samples,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]
    # sample, intermediates = convsample_ddim(model,  steps=args.custom_steps, shape=shape,eta=args.eta)
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    batch = 1024
    with torch.no_grad():
        for i in tqdm(range(int(batch/num_samples)), desc="Generating image samples for cali-data"):
            sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)

            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
        torch.cuda.empty_cache()
    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()

    index = (num_timesteps-1)-t
    all_calib_data = []
    for i in range(int(calib_num_samples/batch)):
        calib_data = None
        t1 = t[i*batch:(i+1)*batch]
        for now_rt, sample_t in enumerate(all_samples):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index

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
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # fix random seed
    seed_everything(opt.seed)
    torch.cuda.set_device(opt.device)
    print(torch.cuda.current_device())

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    if gpu:
        device = "cuda"
    eval_mode = True

    logdir = os.path.join(opt.logdir, "samples", now)
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
    opt.image_folder = imglogdir
    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    logger.info(f"global step: {global_step}")
    logger.info("Switched to EMA weights")
    model.model_ema.store(model.model.parameters())
    model.model_ema.copy_to(model.model)

    args = opt
    if opt.ptq:
        if opt.quant_mode == 'qdiff':
            # wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
            # aq_params = {'n_bits': opt.act_bit, 'symmetric': opt.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': opt.quant_act}
            wq_params = {'n_bits': args.weight_bit, 'symmetric': True, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}
            # with model.ema_scope("Quantizing", restore=False):
            qnn = QuantModel(
                model=model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                sm_abit=opt.sm_abit)
            qnn.cuda()
            qnn.eval()

            print("Setting the first and the last layer to 8-bit")
            qnn.set_first_last_layer_to_8bit()
            qnn.disable_network_output_quantization()
            qnn.set_quant_state(False, False)
            model.model.diffusion_model = qnn

        print("sampling calib data")
        model.model.diffusion_model.set_quant_state(False, False)

        if args.calib_im_mode == "noise_backward_t":
            cali_data = backward_t_calib_data_generator(
                model,
                args,
                args.calib_num_samples,
                args.batch_samples,
                device,
                args.calib_t_mode,
                args.custom_steps,
            )
        elif args.calib_im_mode == "greedy":
            cali_data = backward_featrue_greedy_calib_data_generator(
                model,
                args,
                args.calib_num_samples,
                args.batch_samples,
                device,
                args.custom_steps,
            )
        elif args.calib_im_mode == "Q_diffusion":
            cali_data = Q_diffusion_calib_data_generator(
                model,
                args,
                args.calib_num_samples,
                args.batch_samples,
                device,
                args.calib_t_mode,
                args.custom_steps,
            )
        else:
            raise NotImplementedError
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
                            change_block=False,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False
                            )
            model.model.diffusion_model.set_quant_state(True, True)
            recon_qnn = recon_Qmodel(args, model.model.diffusion_model, cali_data, kwargs)
            model.model.diffusion_model = recon_qnn.recon()

        # model.model.diffusion_model.set_quant_state(False, False)
        model.model.diffusion_model.set_quant_state(True, True)

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        print(sampling_conf)
        logger.info("first_stage_model")
        logger.info(model.first_stage_model)
        logger.info("UNet model")
        logger.info(model.model)


    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, dpm=opt.dpm)

    kwargs = dict(paths=[imglogdir, opt.dataset], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)
    print('FID: ', fid_value)
    print("The para add_loss is {}".format(args.add_loss))
    print("The para lr_w is {}".format(args.lr_w))
    print("The para lr_a is {}".format(args.lr_a))
    logger.info("done.")
