import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import numpy as np
import random

from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from qdiff.quant_layer import QuantModule, lp_loss
from qdiff.quant_block import BaseQuantBlock
from qdiff.quant_model import QuantModel
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.data_utils import save_inp_oup_data

class recon_my_Qmodel():
    def __init__(self, args, iters, model, cali_data):
        self.args = args
        self.iters = iters
        self.model = model
        self.cali_data = cali_data
        self.lr = 4e-5
        self.save_output = {}
        for name, moudle in self.model.named_modules():
            if isinstance(moudle, QuantModule):
                moudle.register_forward_hook(self.hook_fn_forward(name))
            elif isinstance(moudle, BaseQuantBlock):
                moudle.register_forward_hook(self.hook_fn_forward(name))

    def hook_fn_forward(self, name):
        def hook(module, input, output):
            self.save_output[name] = output
        return hook

    def recon(self):
        # model.set_quant_state(False, False)
        cali_data = self.cali_data
        model = self.model
        model.set_quant_state(True, True)
        round_mode = 'learned_hard_sigmoid'

        w_para, a_para = [], []
        def recon_model(module: nn.Module, w_para, a_para, quant_act):
            for name, module in module.named_children():
                if isinstance(module, QuantModule):
                    layer = module
                    '''weight'''
                    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=layer.org_weight.data)
                    layer.weight_quantizer.soft_targets = True
                    w_para += [layer.weight_quantizer.alpha]
                    '''activation'''
                    if quant_act and layer.act_quantizer.delta is not None:
                        layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
                        a_para += [layer.act_quantizer.delta]
                elif isinstance(module, BaseQuantBlock):
                    block = module
                    for module in block.modules():
                        '''weight'''
                        if isinstance(module, QuantModule):
                            if module.split == 0:
                                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data)
                                module.weight_quantizer.soft_targets = True
                                w_para += [module.weight_quantizer.alpha]
                            else :
                                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data[:, :module.split, ...])
                                module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data[:, module.split:, ...])
                                module.weight_quantizer.soft_targets = True
                                module.weight_quantizer_0.soft_targets = True
                                w_para += [module.weight_quantizer.alpha]
                                w_para += [module.weight_quantizer_0.alpha]
                        '''activation'''
                        if isinstance(module, (QuantModule, BaseQuantBlock)):
                            if quant_act and module.act_quantizer.delta is not None:
                                if module.split == 0:
                                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                                    a_para += [module.act_quantizer.delta]
                                else:
                                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                                    module.act_quantizer_0.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_0.delta))
                                    a_para += [module.act_quantizer.delta]
                                    a_para += [module.act_quantizer_0.delta]
                elif name == 'up':# Unet的上采样过程，按顺序重建
                    recon_up_model(module, w_para, a_para, quant_act)
                else:
                    recon_model(module, w_para, a_para, quant_act)
        def recon_up_model(module: nn.Module, w_para, a_para, quant_act):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for up_name, up_module in reversed(list(module.named_children())):
                if isinstance(up_module, QuantModule):
                    layer = up_module
                    '''weight'''
                    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=layer.org_weight.data)
                    layer.weight_quantizer.soft_targets = True
                    w_para += [layer.weight_quantizer.alpha]
                    '''activation'''
                    if quant_act and layer.act_quantizer.delta is not None:
                        layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
                        a_para += [layer.act_quantizer.delta]
                elif isinstance(up_module, BaseQuantBlock):
                    block = up_module
                    for module in block.modules():
                        '''weight'''
                        if isinstance(module, QuantModule):
                            if module.split == 0:
                                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data)
                                module.weight_quantizer.soft_targets = True
                                w_para += [module.weight_quantizer.alpha]
                            else :
                                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data[:, :module.split, ...])
                                module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                                            weight_tensor=module.org_weight.data[:, module.split:, ...])
                                module.weight_quantizer.soft_targets = True
                                module.weight_quantizer_0.soft_targets = True
                                w_para += [module.weight_quantizer.alpha]
                                w_para += [module.weight_quantizer_0.alpha]
                        '''activation'''
                        if isinstance(module, (QuantModule, BaseQuantBlock)):
                            if quant_act and module.act_quantizer.delta is not None:
                                if module.split == 0:
                                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                                    a_para += [module.act_quantizer.delta]
                                else:
                                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                                    module.act_quantizer_0.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_0.delta))
                                    a_para += [module.act_quantizer.delta]
                                    a_para += [module.act_quantizer_0.delta]
                else:
                    recon_model(up_module, w_para, a_para, quant_act)

        recon_model(model, w_para, a_para, self.args.quant_act)
        w_opt, a_opt = None, None
        a_scheduler = None
        if len(w_para) != 0:
            w_opt = torch.optim.Adam(w_para, lr=0.001)
        if len(a_para) != 0:
            a_opt = torch.optim.Adam(a_para, lr=0.05)
            a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=self.iters, eta_min=0.)

        sz = cali_data[0].size(0)
        for i in tqdm(
            range(self.iters), desc=f"recon model {self.iters} iters."
        ):
            idx = random.sample(range(sz), 32)
            w_opt.zero_grad()
            if a_opt:
                a_opt.zero_grad()

            self.save_output.clear()
            t_blocks_out = None
            s_blocks_out = None
            batch_size = 64
            model.set_quant_state(False, False)
            with torch.no_grad():
                # t_res = model(
                #     *[
                #         _[:batch_size].cuda()
                #         for _ in cali_data
                #     ]
                # ) 
                
                t_res = model(
                    *[
                        _[idx].cuda()
                        for _ in cali_data
                    ]
                ) 
                t_blocks_out = self.save_output.copy()
                self.save_output.clear()
            torch.cuda.empty_cache()
            model.set_quant_state(True, True)
            # s_res = model(
            #     *[
            #         _[:batch_size].cuda()
            #         for _ in cali_data
            #     ]
            # ) 
            s_res = model(
                *[
                    _[idx].cuda()
                    for _ in cali_data
                ]
            ) 
            torch.cuda.empty_cache()
            s_blocks_out = self.save_output.copy()
            self.save_output.clear()

        
            for n in t_blocks_out:
                F2 = (F.mse_loss(t_blocks_out[n], torch.zeros(t_blocks_out[n].shape).cuda()))
                t_blocks_out[n]= t_blocks_out[n]/F2
            for n in s_blocks_out:
                F2 = (F.mse_loss(s_blocks_out[n], torch.zeros(s_blocks_out[n].shape).cuda()))
                s_blocks_out[n]= s_blocks_out[n]/F2
            
            L2_loss = torch.zeros(1).cuda()    
            for n in t_blocks_out:
                HWC = t_blocks_out[n].numel()
                # L2_loss += (1.0/HWC)*F.mse_loss(t_blocks_out[n], s_blocks_out[n])
                L2_loss += F.mse_loss(t_blocks_out[n], s_blocks_out[n])
            L2_loss = L2_loss/len(t_blocks_out)
            L1_loss = F.l1_loss(t_res, s_res)

            l = 5.0
            loss = L1_loss + l*L2_loss

            loss.backward(retain_graph=True)

            w_opt.step()
            if a_opt:
                a_opt.step()
            if a_scheduler:
                a_scheduler.step()
            if i%100==0 or i==self.iters-1:
                info1 = f"The {i} iters "
                info2 = f" loss: {loss.item()} "
                info3 = f" L1_loss: {L1_loss} "
                info4 = f" L2_loss: {L2_loss.item()} "
                info = f"{info1},{info2},{info3},{info4}"
                print(info)
        torch.cuda.empty_cache()

        i = 0
        for module in model.modules():
            if isinstance(module, QuantModule):
                '''weight '''
                if module.split == 0:
                    module.weight_quantizer.soft_targets = False
                    i += 1
                else:
                    module.weight_quantizer.soft_targets = False
                    module.weight_quantizer_0.soft_targets = False
                    i += 2
        print(i)
