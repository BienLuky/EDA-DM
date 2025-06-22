import argparse, os, datetime, yaml, sys
sys.path.append('./')
print(sys.path)
import logging
import cv2
import random
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
# from pytorch_lightning import seed_everything
from qdiff.utils import seed_everything
from torch import autocast
from contextlib import nullcontext
import gc

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_control import DDIMSampler_control
from qdiff import QuantModel
from qdiff.utils import AttentionMap
from qdiff_control import set_act_quantize_params_Conditional, set_weight_quantize_params_Conditional, recon_block_Qmodel

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

from task_config import imagenet_get_parser
from calibration import TDAC_imagenet_calib_data_generator

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, device, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.to(device)
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


if __name__ == "__main__":
    parser = imagenet_get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.cuda.set_device(args.device)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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

    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}", device=device)
    model = model.to(device)
    sampler = DDIMSampler_control(model)

    # sample parameters
    batch_size = args.n_batch
    n_rows = args.n_rows if args.n_rows > 0 else args.n_samples
    classes = range(1000)   # define classes to be sampled here
    n_samples_per_class = int(args.n_samples/len(classes))
    xc_all = []
    for class_label in classes:
        xc = torch.tensor(n_samples_per_class*[class_label])
        xc_all.append(xc)
    data = torch.hstack(xc_all)
    data_randperm = torch.randperm(data.size(0))
    data = torch.tensor(data[data_randperm]).to(device)
    args.data = data

    if args.ptq:
        wq_params = {'n_bits': args.weight_bit, 'symmetric': not args.a_sym, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': args.act_bit, 'symmetric': not args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}
        qnn = QuantModel(
            model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
            act_quant_mode="qdiff", sm_abit=args.sm_abit)
        qnn.cuda()
        qnn.eval()

        qnn.set_quant_state(False, False)
        logger.info("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()
        qnn.disable_network_output_quantization()

        if args.no_grad_ckpt:
            logger.info('Not use gradient checkpointing for transformer blocks')
            qnn.set_grad_ckpt(False)

        sampler.model.model.diffusion_model = qnn

        cali_data = TDAC_imagenet_calib_data_generator(
            model,
            args,
            args.calib_num_samples,
            args.batch_samples,
            device,
            args.custom_steps,
        )

        if args.split:
            model.model.diffusion_model.model.split_shortcut = True

        set_weight_quantize_params_Conditional(model, cali_data, args)
        set_act_quantize_params_Conditional(model, cali_data, args)

        if args.recon == True:
            kwargs = dict(cali_data=cali_data, 
                            iters=1000,
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
            sampler.model.model.diffusion_model.set_quant_state(True, True)
            recon_qnn = recon_block_Qmodel(args, sampler.model.model.diffusion_model, cali_data, kwargs)
            sampler.model.model.diffusion_model = recon_qnn.recon()

        sampler.model.model.diffusion_model.set_quant_state(True, True)

    if args.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    logging.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    base_count = 0
    grid_count = 0
    assert(args.cond)

    with torch.no_grad():
        with model.ema_scope():
            all_samples = list()
            for i in tqdm(range(int(args.n_samples/batch_size)), desc="samples"):
                uc = None
                if args.scale != 1.0:                         
                    uc = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
                        )
                xc = data[i*batch_size : (i+1)*batch_size]
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                shape = [3, 64, 64]

                samples_ddim, _ = sampler.sample(S=args.custom_steps,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc, 
                                                eta=args.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_samples_ddim_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                if not args.skip_save:
                    for x_sample in x_samples_ddim_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(imglogdir, f"{base_count:05}.png"))
                        base_count += 1

                if not args.skip_grid:
                    all_samples.append(x_samples_ddim_torch)

        if not args.skip_grid:
            # additionally, save as grid
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=n_rows)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            img = put_watermark(img, wm_encoder)
            img.save(os.path.join(logdir, f'grid-{grid_count:04}.png'))
            grid_count += 1

    logger.info("The para add_loss is {}".format(args.add_loss))
    logger.info("The para lr_w is {}".format(args.lr_w))
    logger.info("The para lr_a is {}".format(args.lr_a))
    logger.info('down!')

