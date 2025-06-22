import torch.nn as nn
from qdiff import QuantModel
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock
from qdiff.block_recon import block_reconstruction
from qdiff.layer_recon import layer_reconstruction
from qdiff.quant_block import QuantResnetBlock, QuantAttnBlock
from qdiff.attn_layer_recon import AttnBlock_layer_reconstruction
import logging
logger = logging.getLogger(__name__)


class recon_layer_Qmodel():
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
                self.recon_block(module.block[0])
                self.recon_block(module.attn[0])
                self.recon_block(module.block[1])
                self.recon_block(module.attn[1])
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
                    self.recon_block(module)
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
                self.recon_block(up_module.block[0])
                self.recon_block(up_module.attn[0])
                self.recon_block(up_module.block[1])
                self.recon_block(up_module.attn[1])
                self.recon_block(up_module.block[2])
                self.recon_block(up_module.attn[2]) 
                layer_reconstruction(self.model, up_module.upsample.conv, **self.kwargs)

            elif isinstance(up_module, QuantModule):
                if up_module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(up_name))
                    loss = layer_reconstruction(self.model, up_module, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name)
            elif isinstance(up_module, BaseQuantBlock):
                if up_module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of block {}'.format(up_name))
                    continue
                else:
                    logger.info('Reconstruction for block {}'.format(up_name))
                    self.recon_block(up_module)
            else:
                self.recon_model(up_module)

    def recon_block(self, block: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        if isinstance(block, QuantResnetBlock):
            self.recon_QuantResnetBlock_block(block)
        elif isinstance(block, QuantAttnBlock):
            self.recon_QuantAttnBlock_block(block)

    def recon_QuantResnetBlock_block(self, module: nn.Module):
        for name, module in module.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    logger.info('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(self.model, module, **self.kwargs)
            else:
                self.recon_QuantResnetBlock_block(module)

    def recon_QuantAttnBlock_block(self, module: nn.Module):
        layer_reconstruction(self.model, module.q, **self.kwargs)
        layer_reconstruction(self.model, module.k, **self.kwargs)
        layer_reconstruction(self.model, module.v, **self.kwargs)
        AttnBlock_layer_reconstruction(self.model, module, **self.kwargs)
        layer_reconstruction(self.model, module.proj_out, **self.kwargs)

    def recon(self):
        self.recon_model(self.model)
        self.model.set_quant_state(weight_quant=True, act_quant=True)
        return self.model
