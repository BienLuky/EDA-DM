import torch
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock, QuantAttnBlock, QuantSMVMatMul, QuantQKMatMul, QuantBasicTransformerBlock
from qdiff.quant_model import QuantModel
from typing import Union
from ldm.models.diffusion.ddim_control import DDIMSampler_control
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

def set_act_quantize_params_Conditional(
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
        if isinstance(m, QuantBasicTransformerBlock):
            m.attn1.act_quantizer_q.set_inited(False)
            m.attn1.act_quantizer_k.set_inited(False)
            m.attn1.act_quantizer_v.set_inited(False)
            m.attn1.act_quantizer_w.set_inited(False)
            m.attn2.act_quantizer_q.set_inited(False)
            m.attn2.act_quantizer_k.set_inited(False)
            m.attn2.act_quantizer_v.set_inited(False)
            m.attn2.act_quantizer_w.set_inited(False)
            
    """set or init step size and zero point in the activation quantizer"""
    batch_size = min(batch_size, cali_data[0].size(0))
    uc = None
    if args.scale != 1.0:                         
        uc = module.get_learned_conditioning(
            {module.cond_stage_key: torch.tensor(batch_size*[1000]).to(module.device)}
            )
    xc = args.data[:batch_size]
    c = module.get_learned_conditioning({module.cond_stage_key: xc.to(module.device)})
    shape = [3, 64, 64]
    
    sampler = DDIMSampler_control(module)

    with torch.no_grad():
        for i in tqdm(range(int(cali_data[0].size(0) / batch_size)), desc="Inited activation"):
        # for i in tqdm(range(int(batch_size / batch_size)), desc="Inited activation"):
            _ = sampler.sample(S=args.custom_steps,
                                conditioning=c,
                                batch_size=batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=args.scale,
                                unconditional_conditioning=uc,
                                eta=args.ddim_eta,
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
        if isinstance(m, QuantBasicTransformerBlock):
            m.attn1.act_quantizer_q.set_inited(True)
            m.attn1.act_quantizer_k.set_inited(True)
            m.attn1.act_quantizer_v.set_inited(True)
            m.attn1.act_quantizer_w.set_inited(True)
            m.attn2.act_quantizer_q.set_inited(True)
            m.attn2.act_quantizer_k.set_inited(True)
            m.attn2.act_quantizer_v.set_inited(True)
            m.attn2.act_quantizer_w.set_inited(True)

def set_weight_quantize_params_Conditional(model, cali_data, args):
    logger.info(f"set_weight_quantize_params")
    model.model.diffusion_model.set_quant_state(True, False)

    for name, module in model.model.diffusion_model.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)

    batch_size = 2
    uc = None
    if args.scale != 1.0:                         
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
            )
    xc = args.data[:batch_size]
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    shape = [3, 64, 64]
    
    sampler = DDIMSampler_control(model)

    with torch.no_grad():
        _ = sampler.sample(S=args.custom_steps,
                            conditioning=c,
                            batch_size=batch_size,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=args.scale,
                            unconditional_conditioning=uc,
                            eta=args.ddim_eta,
                            quant_unet=True, cali_data=[_[:batch_size].cuda() for _ in cali_data])
    torch.cuda.empty_cache()

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            if module.split == 0:
                module.weight_quantizer.set_inited(True)
            else:
                module.weight_quantizer.set_inited(True)
                module.weight_quantizer_0.set_inited(True)