import logging
import torch.nn as nn
from qdiff.quant_block import QuantAttentionBlock
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock

def Change_LDM_model_attnblock(module: nn.Module, act_quant_params: dict = {}):
    for name, child_module in module.named_children():
        if isinstance(child_module, AttentionBlock):
            setattr(module, name, QuantAttentionBlock(child_module, act_quant_params))

        else:
            Change_LDM_model_attnblock(child_module, act_quant_params)


