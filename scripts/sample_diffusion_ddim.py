import argparse, os, glob, datetime, yaml, sys
sys.path.append('./')
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

from qdiff import QuantModel, set_act_quantize_params, set_weight_quantize_params, recon_block_Qmodel, recon_layer_Qmodel
from qdiff.utils import AttentionMap
# from scripts.test import test_fid
logger = logging.getLogger(__name__)

from task_config import cifar_get_parser
from calibration import TDAC_cifar_calib_data_generator


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
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        
        model.to(self.device)
        model.eval()
        if self.args.ptq:
            wq_params = {'n_bits': args.weight_bit, 'symmetric': not args.a_sym, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {'n_bits': args.act_bit, 'symmetric': not args.a_sym, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.quant_act, "prob": 0.5}

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
        
    def sample_fid(self, model):
        config = self.config
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
        return x


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = cifar_get_parser()
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
    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")

    diffusion = Diffusion(args, config)
    qnn = diffusion.QModel()
    if args.ptq:
        logger.info("Setting the first and the last layer to 8-bit")
        qnn.set_first_last_layer_to_8bit()
        qnn.disable_network_output_quantization()

        logger.info("sampling calib data")
        qnn.set_quant_state(False, False)
        cali_data = TDAC_cifar_calib_data_generator(
                qnn.model,
                config,
                args.lamda,
                args.calib_num_samples,
                args.batch_samples,
                "cuda",
                diffusion,
                args.class_cond,
            )

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
                            recon_w=True,
                            recon_a=True
                            )

            qnn.set_quant_state(True, True)

            if args.block_recon:
                qnn.block_loss = []
                recon_qnn = recon_block_Qmodel(args, qnn, cali_data, kwargs)
                qnn = recon_qnn.recon()
            elif args.layer_recon:
                recon_qnn = recon_layer_Qmodel(args, qnn, cali_data, kwargs)
                qnn = recon_qnn.recon()
            else:
                raise NotImplementedError

        qnn.set_quant_state(True, True)

    diffusion.sample_fid(qnn)
    # test_fid(args.image_folder, device=args.device)
    logger.info("The para add_loss is {}".format(args.add_loss))
    logger.info("The para lr_w is {}".format(args.lr_w))
    logger.info("The para lr_a is {}".format(args.lr_a))

