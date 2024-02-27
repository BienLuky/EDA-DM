import logging
import torch.nn as nn
from qdiff.quant_block import BaseQuantBlock, QuantResnetBlock, Quantdownsample_resnet_down, QuantAttnBlock, Quantdownsample_attn_down, Quantupsample_resnet_down, Quantupsample_attn_down
from qdiff.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from ddim.models.diffusion import Upsample, Downsample

logger = logging.getLogger(__name__)


class Change_model_block(nn.Module):

    def __init__(self, model: nn.Module, act_quant_params: dict = {}):
        super().__init__()
        self.model = model
        self.last_layer = None#记录上一个layer是什么
        self.last_layer_name = None#记录上一个layer是什么
        # self.last_resblock_layer = None#
        # self.last_resblock_layer_name = None#
        self.skip_layer_name=[]
        
        self.get_up_down_prelayer(self.model)
        self.change_up_down_block(self.model, act_quant_params)

    def get_up_down_prelayer(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, BaseQuantBlock):
                self.last_layer = module
                self.last_layer_name = name
                # if isinstance(module, QuantResnetBlock):
                #     self.last_resblock_layer = module
                #     self.last_resblock_layer_name = name
                # print(name)
            elif isinstance(module, (Downsample, Upsample)):
            # elif isinstance(module, Upsample):
                module.last_layer = self.last_layer
                # module.last_resblock_layer = self.last_resblock_layer
                self.skip_layer_name.append(self.last_layer_name)
                # self.skip_layer_name.append(self.last_resblock_layer_name)
                # print(name)
        self.last_layer = None
        self.last_layer_name = None
        # self.last_resblock_layer = None
        # self.last_resblock_layer_name = None

    def set_recon_state(self):
        for name, module in self.model.named_modules():
            if isinstance(module, BaseQuantBlock) and name in self.skip_layer_name:
                module.can_recon = False

    def change_up_down_block(self, module: nn.Module, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if isinstance(child_module, Downsample):
                if isinstance(child_module.last_layer, QuantResnetBlock):
                    setattr(module, name, Quantdownsample_resnet_down(child_module.last_layer, child_module, act_quant_params))
                elif isinstance(child_module.last_layer, QuantAttnBlock):
                    setattr(module, name, Quantdownsample_attn_down(child_module.last_layer, child_module, act_quant_params))

            elif isinstance(child_module, Upsample):
                if isinstance(child_module.last_layer, QuantResnetBlock):
                    setattr(module, name, Quantupsample_resnet_down(child_module.last_layer, child_module, act_quant_params))
                elif isinstance(child_module.last_layer, QuantAttnBlock):
                    # setattr(module, name, Quantupsample_attn_down(child_module.last_resblock_layer, child_module.last_layer, child_module, act_quant_params))  
                    setattr(module, name, Quantupsample_attn_down(child_module.last_layer, child_module, act_quant_params))         

            elif isinstance(child_module, (QuantModule, BaseQuantBlock)):
                continue
            #     setattr(module, name, QuantUpsample(preblock, downsample))

            else:
                self.change_up_down_block(child_module, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

