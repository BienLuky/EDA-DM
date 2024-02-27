import argparse, os, datetime, yaml, sys
sys.path.append('./pytorch-fid/src')
sys.path.append('./clip-score/src')
print(sys.path)
import logging
import cv2
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
import random
# from pytorch_lightning import seed_everything
from qdiff.utils import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from qdiff import QuantModel
from qdiff.utils import resume_cali_model, AttentionMap
from qdiff_control import set_act_quantize_params_Stable, set_weight_quantize_params_Stable, get_prompts, recon_Qmodel, center_resize_image
from pytorch_fid.fid_score import calculate_fid_given_paths
from clip_score.clip_score import calculate_clip_given_paths
# from thop.profile import profile

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


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


def load_model_from_config(config, ckpt, verbose=False):
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

    # device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
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


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

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
    plt.savefig('./COCO_PTQ_t_num.png')

    uc = None
    if args.scale != 1.0:
        uc = model.get_learned_conditioning(calib_num_samples * [""])
    prompts = args.list_prompts[:calib_num_samples]
    c = model.get_learned_conditioning(prompts)
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    samples = []
    cond = []
    uncond = []
    ts = None

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            # sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c[i*num_samples:(i+1)*num_samples],
                                            batch_size=num_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                            eta=args.ddim_eta,
                                            x_T=start_code)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
            ts_next = intermediates['ts_next']
            cond.append(intermediates['cond'][0].to(device))
            uncond.append(intermediates['uncond'][0].to(device))
        torch.cuda.empty_cache()
    cond = torch.cat(cond)
    uncond = torch.cat(uncond)
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
    calib_t_next = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
        calib_t_next.append(ts_next[time][0].to(device))
    t = torch.tensor(calib_t).to(device)
    t_next = torch.tensor(calib_t_next).to(device)

    return calib_data, t, index, cond, uncond, t_next

def backward_featrue_greedy_calib_data_generator(
    model, args, calib_num_samples, num_samples, device, num_timesteps
):

    uc = None
    if args.scale != 1.0:
        uc = model.get_learned_conditioning(calib_num_samples * [""])
    prompts = args.list_prompts[:calib_num_samples]
    c = model.get_learned_conditioning(prompts)
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    samples = []
    cond = []
    uncond = []
    ts = None

    hooks = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                _, intermediates, feature_map = sampler.sample(S=args.ddim_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta,
                                                x_T=start_code,
                                                hooks=hooks)
            else:
                _, intermediates = sampler.sample(S=args.ddim_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta,
                                                x_T=start_code)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
            ts_next = intermediates['ts_next']
            cond.append(intermediates['cond'][0].to(device))
            uncond.append(intermediates['uncond'][0].to(device))
        torch.cuda.empty_cache()
    cond = torch.cat(cond)
    uncond = torch.cat(uncond)
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
    plt.savefig('./COCO_dense_num.png')
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
    plt.savefig('./COCO_Cos_dis.png')
    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    # l = 1.0
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    x = range(len(w))
    f = plt.figure()
    plt.plot(x, w.to('cpu'))
    plt.savefig('./COCO_w.png')

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
            if t_num[i] > 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./COCO_TDAC_t_num.png')

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
    calib_t_next = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
        calib_t_next.append(ts_next[time][0].to(device))
    t = torch.tensor(calib_t).to(device)
    t_next = torch.tensor(calib_t_next).to(device)

    return calib_data, t, index, cond, uncond, t_next


def Q_diffusion_calib_data_generator(
    model, args, calib_num_samples, num_samples, device, t_mode, num_timesteps
):

    t = generate_t(t_mode, calib_num_samples, num_timesteps, device).long()

    uc = None
    if args.scale != 1.0:
        uc = model.get_learned_conditioning(calib_num_samples * [""])
    prompts = args.list_prompts[:calib_num_samples]
    c = model.get_learned_conditioning(prompts)
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    samples = []
    cond = []
    uncond = []
    ts = None

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            # sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)
            _, intermediates = sampler.sample(S=args.ddim_steps,
                                            conditioning=c[i*num_samples:(i+1)*num_samples],
                                            batch_size=num_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=args.scale,
                                            unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                            eta=args.ddim_eta,
                                            x_T=start_code)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
            ts_next = intermediates['ts_next']
            cond.append(intermediates['cond'][0].to(device))
            uncond.append(intermediates['uncond'][0].to(device))
        torch.cuda.empty_cache()
    cond = torch.cat(cond)
    uncond = torch.cat(uncond)
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
    calib_t_next = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
        calib_t_next.append(ts_next[time][0].to(device))
    t = torch.tensor(calib_t).to(device)
    t_next = torch.tensor(calib_t_next).to(device)

    return calib_data, t, index, cond, uncond, t_next

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
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
        default="/dataset/coco2014/val2014_resize"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_batch",
        type=int,
        default=2,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
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
        choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use"
    )

    # qdiff specific configs
    parser.add_argument(
        "--device", type=str,
        default="cuda:0",
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
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
        choices=['normal', 'Qdiff'],
    )
    parser.add_argument(
        "--calib_num_samples",
        default=256,
        type=int,
        help="size of the calibration dataset",
    )
    parser.add_argument(
        "--batch_samples",
        default=4,
        type=int,
        help="size of the sample dataset",
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
    parser.add_argument(
        "--lamda",
        type=float,
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)
    torch.cuda.set_device(opt.device)
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)


    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    batch_size = opt.n_batch
    n_rows = opt.n_rows if opt.n_rows > 0 else opt.n_samples
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [opt.n_samples * [prompt]]

    else:
        logging.info(f"reading prompts from {opt.from_file}")
        opt.list_prompts = get_prompts(opt.from_file)
        data = opt.list_prompts[:opt.n_samples]

    base_count = 0
    prompt_path = os.path.join(logdir, "image_prompts")
    os.makedirs(prompt_path)

    for prompt in data:
        name = os.path.join(prompt_path, f"{base_count:05}.txt")
        # name = prompt_path + f"{base_count:05}.txt"
        file = open(name, 'w')
        file.write(prompt)
        file.close()
        base_count = base_count + 1
    assert(opt.cond)

    args = opt
    if opt.ptq:
        if opt.quant_mode == 'qdiff':
            # wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
            # aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'max', 'leaf_param':  opt.quant_act}
            wq_params = {'n_bits': opt.weight_bit, 'symmetric': True, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {'n_bits': opt.act_bit, 'symmetric': True, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': opt.quant_act, "prob": 0.5}
            qnn = QuantModel(
                model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                act_quant_mode="qdiff", sm_abit=opt.sm_abit)#
            qnn.cuda()
            qnn.eval()

            qnn.set_quant_state(False, False)
            print("Setting the first and the last layer to 8-bit")
            qnn.set_first_last_layer_to_8bit()
            qnn.disable_network_output_quantization()

            if opt.no_grad_ckpt:
                logger.info('Not use gradient checkpointing for transformer blocks')
                qnn.set_grad_ckpt(False)

            sampler.model.model.diffusion_model = qnn

        args.custom_steps = args.ddim_steps
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
        if opt.split:
            setattr(sampler.model.model.diffusion_model, "split", True)

        set_weight_quantize_params_Stable(model, cali_data, args)
        set_act_quantize_params_Stable(model, cali_data, args)

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
                            batch_size=2,
                            input_prob=0.5,
                            add_loss=args.add_loss,
                            change_block=False,
                            recon_w=True,
                            recon_a=True,
                            keep_gpu=False
                            )
            sampler.model.model.diffusion_model.set_quant_state(True, True)
            recon_qnn = recon_Qmodel(args, sampler.model.model.diffusion_model, cali_data, kwargs)
            sampler.model.model.diffusion_model = recon_qnn.recon()

        # sampler.model.model.diffusion_model.set_quant_state(False, False)
        sampler.model.model.diffusion_model.set_quant_state(True, True)

    logging.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    base_count = 0
    grid_count = 0

    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for i in tqdm(range(int(opt.n_samples/batch_size)), desc="samples"):
                        prompts = data[i*batch_size : (i+1)*batch_size]
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(imglogdir, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                if not opt.skip_grid:
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
                toc = time.time()

    image_resize = os.path.join(logdir, "image_resize")
    os.makedirs(image_resize)

    center_resize_image(imglogdir, image_resize, (300, 300))
    kwargs = dict(paths=[image_resize, args.dataset], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)

    kwargs = dict(real_path=imglogdir, 
                    fake_path=prompt_path,
                    batch_size=100,
                    device=device, 
                    clip_model="ViT-L/14",
                    num_workers=8
                    )
    clip_score = calculate_clip_given_paths(**kwargs)
    print('FID: ', fid_value)
    print('CLIP Score: ', clip_score)
    print("The para add_loss is {}".format(args.add_loss))
    print("The para lr_w is {}".format(args.lr_w))
    print("The para lr_a is {}".format(args.lr_a))

if __name__ == "__main__":
    main()
