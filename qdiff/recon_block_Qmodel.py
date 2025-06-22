import torch.nn as nn
from qdiff import QuantModel
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock, QuantAttentionBlock
from qdiff.block_recon import block_reconstruction
from qdiff.layer_recon import layer_reconstruction
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock
import logging
logger = logging.getLogger(__name__)

def Change_LDM_model_attnblock(module: nn.Module, act_quant_params: dict = {}):
    for name, child_module in module.named_children():
        if isinstance(child_module, AttentionBlock):
            setattr(module, name, QuantAttentionBlock(child_module, act_quant_params))
        else:
            Change_LDM_model_attnblock(child_module, act_quant_params)

class recon_block_Qmodel():
    def __init__(self, args, qnn, cali_data, kwargs):
        self.args = args
        self.model = qnn
        self.cali_data = cali_data
        self.kwargs = kwargs
        self.down_name = None

    def recon_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == '1' and isinstance(module, BaseQuantBlock) == 0:
                logger.info('reconstruction for down 1 modulelist')
                loss = block_reconstruction(self.model, module.block[0], **self.kwargs)
                loss = block_reconstruction(self.model, module.attn[0], **self.kwargs)
                loss = block_reconstruction(self.model, module.block[1], **self.kwargs)
                block_reconstruction(self.model, module.attn[1], **self.kwargs)
                layer_reconstruction(self.model, module.downsample.conv, **self.kwargs)
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    loss = layer_reconstruction(self.model, module, **self.kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(name))
                    loss = block_reconstruction(self.model, module, **self.kwargs)
            elif name == 'up':
                self.recon_up_model(module)
            else:
                self.recon_model(module)

    def recon_up_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for up_name, up_module in reversed(list(module.named_children())):
            if up_name == '1':
                logger.info('reconstruction for up 1 modulelist')
                loss = block_reconstruction(self.model, up_module.block[0], **self.kwargs)
                loss = block_reconstruction(self.model, up_module.attn[0], **self.kwargs)
                loss = block_reconstruction(self.model, up_module.block[1], **self.kwargs)
                loss = block_reconstruction(self.model, up_module.attn[1], **self.kwargs)
                loss = block_reconstruction(self.model, up_module.block[2], **self.kwargs)
                loss = block_reconstruction(self.model, up_module.attn[2], **self.kwargs) 
                loss = layer_reconstruction(self.model, up_module.upsample.conv, **self.kwargs)
            elif isinstance(up_module, QuantModule):
                if up_module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(up_name))
                    loss = layer_reconstruction(self.model, up_module, **self.kwargs)
            elif isinstance(up_module, BaseQuantBlock):
                if up_module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of block {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(up_name))
                    loss = block_reconstruction(self.model, up_module, **self.kwargs)
            else:
                self.recon_model(up_module)
            
    def recon(self):
        self.recon_model(self.model)
        self.model.set_quant_state(weight_quant=True, act_quant=True)
        return self.model
