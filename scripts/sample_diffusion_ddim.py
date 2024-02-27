import argparse, os, glob, datetime, yaml, sys
print(sys.path)
import logging
import math
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda import amp
# from pytorch_lightning import seed_everything
from qdiff.utils import seed_everything

from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from ddim.functions.denoising import generalized_steps, cali_generalized_steps

import torchvision.utils as tvu

from qdiff import QuantModel, set_act_quantize_params, set_weight_quantize_params, recon_Qmodel, recon_my_Qmodel, new_recon_Qmodel, Change_model_block, recon_layer_Qmodel
from qdiff.utils import resume_cali_model, AttentionMap
from scripts.test import test_fid
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
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
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
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
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument("--change_block_recon", action="store_true",
        help="change_block_recon"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--calib_im_mode",
        default="random",
        type=str,
        choices=["random", "raw", "raw_forward_t", "noise_backward_t", "greedy", "Q_diffusion"],
    )
    parser.add_argument(
        "--calib_t_mode",
        default="random",
        type=str,
        choices=["random", "1", "-1", "mean", "uniform" , 'manual' ,'normal' ,'poisson', 'Qdiff'],
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
        "--block_recon", action="store_true",
        help="use reconstruction"
    )
    parser.add_argument(
        "--layer_recon", action="store_true",
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
        default=1.2
    )
    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def generate_t(t_mode, num_samples, diffusion, device):
    if t_mode == "1":
        t = torch.tensor([1] * num_samples, device=device)  # TODO timestep gen
    elif t_mode == "normal":
        shape = torch.Tensor(num_samples)
        normal_val = torch.nn.init.normal_(shape, mean=0.4, std=0.4)*diffusion.num_timesteps
        x_min, x_max = torch._aminmax(normal_val)
        normal_val = (normal_val-x_min)/(x_max-x_min)*diffusion.num_timesteps
        t = normal_val.clone().type(torch.int).to(device=device)
    elif t_mode == "Qdiff":
        t = []
        vert = 5
        steps = diffusion.num_timesteps/vert
        num = int(num_samples/steps)
        for time in range(diffusion.num_timesteps):
            if time%vert==0 and time!=(diffusion.num_timesteps-1):
                t.append(torch.full((num, ), time).int())
            elif time==(diffusion.num_timesteps-1):
                t.append(torch.full((num_samples - int(num*steps), ), time).int())
        t = torch.hstack(t).to(device=device)
        idx = random.sample(range(len(t)), len(t))
        t = t[idx].to(device)
    else:
        raise NotImplementedError
    return t.clamp(0, diffusion.num_timesteps - 1)


def backward_t_calib_data_generator(
    model, config, calib_num_samples, num_samples, device, t_mode, diffusion, class_cond=True
):
    model_kwargs = {}
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        model_kwargs["y"] = cls
    t = generate_t(t_mode, calib_num_samples, diffusion, device).long()

    PTQ_t_num = torch.zeros(diffusion.num_timesteps)
    for i in t:
        PTQ_t_num[i] += 1
    # t, index = t.sort()
    x = range(len(PTQ_t_num))
    f = plt.figure()
    plt.plot(x, PTQ_t_num.to('cpu'))
    plt.savefig('./PTQ_t_num.png')

    loop_fn = (
        cali_generalized_steps

    )
    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    img = torch.randn(*shape, device=device) 
    seq = diffusion.seq
    all_sample=[]

    calib_data = None
    all_mask = []
    for now_rt, sample_t in enumerate(
            loop_fn(
                model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args
            )
        ):
        if len(seq)==now_rt:
            break
        else:
            sample_t = sample_t[-1]
            all_sample.append(sample_t)
    torch.cuda.empty_cache()

    all_calib_data = []
    all_mask=[]
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(seq[(len(seq)-1)-time])
    t = torch.tensor(calib_t).to(device)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t

def backward_featrue_greedy_calib_data_generator(
    model, config, lamda, calib_num_samples, num_samples, device, diffusion, class_cond=True
):
    model_kwargs = {}
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        model_kwargs["y"] = cls

    loop_fn = (
        cali_generalized_steps
    )

    seq = diffusion.seq
    all_sample=[]
    calib_data = None
    all_mask = []
    hooks = []
    feature_map = []
    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    img = torch.randn(*shape, device=device) 
    hooks.append(AttentionMap(model.mid.attn_1))
    for now_rt, sample_t in enumerate(
            loop_fn(
                model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args
            )
        ):
        if len(seq)==now_rt:
            all_sample = sample_t[:-1]
            feature_map.append(hooks[0].feature[0])
            break
        elif now_rt==0:
            continue
        else:
            feature_map.append(hooks[0].feature[0])

    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    dense_r = 3.0
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                # print(mse)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1
    x = range(len(dense_num))
    f = plt.figure(figsize=(8, 6), dpi=240)
    plt.plot(x, dense_num)
    plt.savefig('./CIFAR_dense_num.png')
    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())

    calib_mask = []
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    # 初始化S
    index = torch.argmax(dense_num_normal)
    calib_mask.append(index)

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))
    x = range(len(Cos_dis))
    f = plt.figure(figsize=(8, 6), dpi=240)
    plt.plot(x, Cos_dis.to('cpu'))
    plt.savefig('./CIFAR_Cos_dis.png')
    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    x = range(len(w))
    f = plt.figure()
    plt.plot(x, w.to('cpu'))
    plt.savefig('./CIFAR_w.png')

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    # else:
    #     t_num_add = t_num_sort[t_error:]
    #     t_num[t_num_add] -= 1
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
    plt.savefig('./CIFAR_TDAC_t_num.png')

    assert torch.sum(t_num)==calib_num_samples
    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t).to(device)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)

    all_calib_data = []
    all_mask=[]
    # '''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)
    # '''
    '''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        if i != 0:
            img = torch.randn(*shape, device=device)
            all_sample = generalized_steps(model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args)
            all_sample = all_sample[0][:-1]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            calib_data = calib_data.to(device)
            sample_t = sample_t.to(device)
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)
    '''

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(seq[(len(seq)-1)-time])
    t = torch.tensor(calib_t).to(device)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t

def Q_diffusion_calib_data_generator(
    model, config, calib_num_samples, num_samples, device, t_mode, diffusion, class_cond=True
):
    model_kwargs = {}
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        model_kwargs["y"] = cls
    t = generate_t(t_mode, calib_num_samples, diffusion, device).long()

    loop_fn = (
        cali_generalized_steps
    )
    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    img = torch.randn(*shape, device=device) 
    seq = diffusion.seq

    all_sample=[]
    for now_rt, sample_t in enumerate(
            loop_fn(
                model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args
            )
        ):
        if len(seq)==now_rt:
            all_sample = sample_t[:-1]
            break
    torch.cuda.empty_cache()

    all_calib_data = []
    all_mask=[]
    #'''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)
    #'''
    '''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        if i != 0:
            img = torch.randn(*shape, device=device)
            all_sample = generalized_steps(model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args)
            all_sample = all_sample[0][:-1]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            calib_data = calib_data.to(device)
            sample_t = sample_t.to(device)
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)
    '''
    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(seq[(len(seq)-1)-time])
    t = torch.tensor(calib_t).to(device)

    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.model = Model(self.config)
        self.num_timesteps = args.timesteps
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_diffusion_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        if self.args.skip_type == "uniform":
            skip = self.num_diffusion_timesteps // self.args.timesteps
            seq = range(0, self.num_diffusion_timesteps, skip)
            self.seq = seq
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_diffusion_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
            self.seq = seq
        else:
            raise NotImplementedError

    def QModel(self):
        model = self.model
        # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        # ckpt = "/home/liuxuewen/Dome/q-diffusion/rscratch/xiuyu/.cache/diffusion_models_converted/ema_diffusion_cifar10_model/model-790000.ckpt"
        # logger.info("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        if self.args.ptq:
            if self.args.quant_mode == 'qdiff':
                # wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
                # aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act}
                wq_params = {'n_bits': args.weight_bit, 'symmetric': True, 'channel_wise': True, 'scale_method': 'mse'}
                aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}
                # wq_params = {'n_bits': args.weight_bit, 'symmetric': True, 'channel_wise': True, 'scale_method': 'max'}
                # aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act}
                qnn = QuantModel(
                    model=model, weight_quant_params=wq_params, act_quant_params=aq_params, 
                    sm_abit=self.args.sm_abit)
                qnn.to(self.device)
                qnn.eval()
                model = qnn

        model.to(self.device)
        if self.args.verbose:
            logger.info("quantized model")
            logger.info(model)

        model.eval()
        return model
        # self.sample_fid(model)
        

    def sample_fid(self, model):
        config = self.config
        # img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        img_id = 0
        logger.info(f"starting from image {img_id}")
        total_n_samples = self.args.max_images
        n_rounds = math.ceil((total_n_samples - img_id) / config.sampling.batch_size)

        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image(x, model)
                x = inverse_data_transform(config, x)

                if img_id + x.shape[0] > self.args.max_images:
                    assert(i == n_rounds - 1)
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                    )
                    img_id += 1
                torch.cuda.empty_cache()
    def ddim_generalized_steps(self, model, x, seq):
        
        betas = self.betas
        xs = generalized_steps(
            x, seq, model, betas, eta=self.args.eta, args=self.args)
        x = xs[0][:-1]
        return x
    def sample_image(self, x, model, last=True):
        
        seq = self.seq
        betas = self.betas
        xs = generalized_steps(
            x, seq, model, betas, eta=self.args.eta, args=self.args)
        x = xs
        if last:
            x = x[0][-1]
            # x = x[0][40]
        return x

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)
    torch.cuda.set_device(args.device)

    # setup logger
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
    # args.image_folder = "/home/liuxuewen/Dome/q-diffusion/image"
    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    diffusion = Diffusion(args, config)
    qnn = diffusion.QModel()
    print("Setting the first and the last layer to 8-bit")
    qnn.set_first_last_layer_to_8bit()
    qnn.disable_network_output_quantization()

    print("sampling calib data")
    qnn.set_quant_state(False, False)
    if args.calib_im_mode == "noise_backward_t":
        cali_data = backward_t_calib_data_generator(
            qnn.model,
            config,
            args.calib_num_samples,
            args.batch_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    elif args.calib_im_mode == "greedy":
        cali_data = backward_featrue_greedy_calib_data_generator(
            qnn.model,
            config,
            args.lamda,
            args.calib_num_samples,
            args.batch_samples,
            "cuda",
            diffusion,
            args.class_cond,
        )
    elif args.calib_im_mode == "Q_diffusion":
        cali_data = Q_diffusion_calib_data_generator(
            qnn.model,
            config,
            args.calib_num_samples,
            args.batch_samples,
            "cuda",
            args.calib_t_mode,
            diffusion,
            args.class_cond,
        )
    else:
        raise NotImplementedError

    if args.split == True:
        qnn.model.config.split_shortcut = True
    
    set_weight_quantize_params(qnn, cali_data)
    set_act_quantize_params(qnn, cali_data)
    if args.recon == True:
        kwargs = dict(cali_data=cali_data, 
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
                        change_block=args.change_block_recon,
                        recon_w=True,
                        recon_a=True
                        )

        qnn.set_quant_state(True, True)

        if args.change_block_recon == True:
            print("change_block_recon")
            aq_params = {'n_bits': args.act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}
            qnn = Change_model_block(qnn, act_quant_params=aq_params)
            qnn.set_recon_state()
            qnn = qnn.model
            qnn.model.config.change_block_recon = True

        if args.block_recon:
            qnn.block_loss = []
            recon_qnn = recon_Qmodel(args, qnn, cali_data, kwargs)
            # recon_qnn = new_recon_Qmodel(args, qnn, **kwargs)
            qnn = recon_qnn.recon()
            # torch.save(qnn.block_loss, "all_block_moudle_loss.pt")
        elif args.layer_recon:
            recon_qnn = recon_layer_Qmodel(args, qnn, cali_data, kwargs)
            qnn = recon_qnn.recon()
        else:
            raise NotImplementedError

    qnn.set_quant_state(True, True)
    # qnn.set_quant_state(False, False)
    diffusion.sample_fid(qnn)
    test_fid(args.image_folder, device=args.device)
    print("The para add_loss is {}".format(args.add_loss))
    print("The para lr_w is {}".format(args.lr_w))
    print("The para lr_a is {}".format(args.lr_a))

