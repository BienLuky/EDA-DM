import torch.nn as nn
from qdiff import QuantModel
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock
from qdiff.block_recon import block_reconstruction
from qdiff.layer_recon import layer_reconstruction


class recon_Qmodel():
    def __init__(self, args, qnn, cali_data, kwargs):
        self.args = args
        self.model = qnn
        self.cali_data = cali_data
        self.kwargs = kwargs
        self.down_name = None
        self.layer_loss = []
        self.layer_name = []

    def recon_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == '1' and isinstance(module, BaseQuantBlock) == 0:
                print('reconstruction for down 1 modulelist')
                loss = block_reconstruction(self.model, module.block[0], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(name+"block_0")
                loss = block_reconstruction(self.model, module.attn[0], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(name+"attn_0")
                loss = block_reconstruction(self.model, module.block[1], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(name+"block_1")
                if self.args.change_block_recon == True:
                    block_reconstruction(self.model, module.downsample, **self.kwargs)
                else:
                    block_reconstruction(self.model, module.attn[1], **self.kwargs)
                    layer_reconstruction(self.model, module.downsample.conv, **self.kwargs)
                # loss = block_reconstruction(self.model, module.attn[1], **self.kwargs)
                # self.layer_loss.append(loss)
                # self.layer_name.append(name+"attn_1")
                # loss = layer_reconstruction(self.model, module.downsample.conv, **self.kwargs)
                # self.layer_loss.append(loss)
                # self.layer_name.append(name+"down_conv")
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    loss = layer_reconstruction(self.model, module, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(name)
            elif isinstance(module, BaseQuantBlock):
                if module.can_recon == False:
                    continue
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    loss = block_reconstruction(self.model, module, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(name)
            elif name == 'up':# Unet的上采样过程，按顺序重建
                self.recon_up_model(module)
            else:
                self.recon_model(module)
    def recon_up_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for up_name, up_module in reversed(list(module.named_children())):
            if up_name == '1':
                print('reconstruction for up 1 modulelist')
                loss = block_reconstruction(self.model, up_module.block[0], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(up_name+"block_0")
                loss = block_reconstruction(self.model, up_module.attn[0], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(up_name+"attn_0")
                loss = block_reconstruction(self.model, up_module.block[1], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(up_name+"block_1")
                loss = block_reconstruction(self.model, up_module.attn[1], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(up_name+"attn_1")
                loss = block_reconstruction(self.model, up_module.block[2], **self.kwargs)
                self.layer_loss.append(loss)
                self.layer_name.append(up_name+"block_2")
                if self.args.change_block_recon == True:
                    loss = block_reconstruction(self.model, up_module.upsample, **self.kwargs) 
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name+"up_conv")
                else:
                    loss = block_reconstruction(self.model, up_module.attn[2], **self.kwargs) 
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name+"attn_2")
                    loss = layer_reconstruction(self.model, up_module.upsample.conv, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name+"up_conv")
            elif isinstance(up_module, QuantModule):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(up_name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(up_name))
                    loss = layer_reconstruction(self.model, up_module, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name)
            elif isinstance(up_module, BaseQuantBlock):
                if up_module.can_recon == False:
                    continue
                if up_module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(up_name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(up_name))
                    loss = block_reconstruction(self.model, up_module, **self.kwargs)
                    self.layer_loss.append(loss)
                    self.layer_name.append(up_name)
            else:
                self.recon_model(up_module)
            
    def recon(self):
        self.recon_model(self.model)
        self.model.set_quant_state(weight_quant=True, act_quant=True)
        return self.model
