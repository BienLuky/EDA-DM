import torch
from tqdm import tqdm
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock, QuantAttnBlock, QuantSMVMatMul, QuantQKMatMul
from .quant_model import QuantModel
from typing import Union
from ldm.models.diffusion.ddim import DDIMSampler
import logging
logger = logging.getLogger(__name__)

def set_act_quantize_params_LDM(
    module,
    cali_data,
    args,
    batch_size: int = 32,
):
    logger.info(f"set_act_quantize_params")
    module.model.diffusion_model.set_quant_state(True, True)

    for m in module.model.diffusion_model.modules():
        if isinstance(m, QuantModule):
            if m.split == 0:
                m.act_quantizer.set_inited(False)
            else:
                m.act_quantizer.set_inited(False)
                m.act_quantizer_0.set_inited(False)
        if isinstance(m, QuantAttnBlock):
            m.act_quantizer_k.set_inited(False)
            m.act_quantizer_q.set_inited(False)
            m.act_quantizer_v.set_inited(False)
            m.act_quantizer_w.set_inited(False)
        if isinstance(m, QuantSMVMatMul):
            m.act_quantizer_v.set_inited(False)
            m.act_quantizer_w.set_inited(False)
        if isinstance(m, QuantQKMatMul):
            m.act_quantizer_k.set_inited(False)
            m.act_quantizer_q.set_inited(False)
            
    """set or init step size and zero point in the activation quantizer"""
    batch_size = min(batch_size, cali_data[0].size(0))
    shape = [batch_size,
             module.model.diffusion_model.in_channels,
             module.model.diffusion_model.image_size,
             module.model.diffusion_model.image_size]
    ddim = DDIMSampler(module)
    bs = shape[0]
    shape = shape[1:]

    with torch.no_grad():
        for i in tqdm(range(int(cali_data[0].size(0) / batch_size)), desc="Inited activation"):
        # for i in tqdm(range(int(batch_size / batch_size)), desc="Inited activation"):
            sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False, 
                                                quant_unet=True, cali_data=[_[i * batch_size : (i + 1) * batch_size].cuda() for _ in cali_data])
    torch.cuda.empty_cache()

    for m in module.modules():
        if isinstance(m, QuantModule):
            if m.split == 0:
                m.act_quantizer.set_inited(True)
            else:
                m.act_quantizer.set_inited(True)
                m.act_quantizer_0.set_inited(True)
        if isinstance(m, QuantAttnBlock):
            m.act_quantizer_k.set_inited(True)
            m.act_quantizer_q.set_inited(True)
            m.act_quantizer_v.set_inited(True)
            m.act_quantizer_w.set_inited(True)
        if isinstance(m, QuantSMVMatMul):
            m.act_quantizer_v.set_inited(True)
            m.act_quantizer_w.set_inited(True)
        if isinstance(m, QuantQKMatMul):
            m.act_quantizer_k.set_inited(True)
            m.act_quantizer_q.set_inited(True)

def set_weight_quantize_params_LDM(model, cali_data, args):
    logger.info(f"set_weight_quantize_params")
    model.model.diffusion_model.set_quant_state(True, False)

    for name, module in model.model.diffusion_model.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)

    batch_size = 8
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]

    with torch.no_grad():
        sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False, 
                                            quant_unet=True, cali_data=[_[:batch_size].cuda() for _ in cali_data])
    torch.cuda.empty_cache()

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            if module.split == 0:
                module.weight_quantizer.set_inited(True)
            else:
                module.weight_quantizer.set_inited(True)
                module.weight_quantizer_0.set_inited(True)