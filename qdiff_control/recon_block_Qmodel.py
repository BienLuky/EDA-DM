import torch.nn as nn
from qdiff import QuantModel
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock
from qdiff_control.block_recon import block_reconstruction
from qdiff_control.layer_recon import layer_reconstruction
import logging
logger = logging.getLogger(__name__)

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
            if isinstance(module, QuantModule):
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
            else:
                self.recon_model(module)
            
    def recon(self):
        self.recon_model(self.model)
        self.model.set_quant_state(weight_quant=True, act_quant=True)
        return self.model
