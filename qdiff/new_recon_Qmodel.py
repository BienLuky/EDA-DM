import torch
import torch.nn as nn
import random
from qdiff import QuantModel
from qdiff.quant_layer import QuantModule, lp_loss
from qdiff.quant_block import BaseQuantBlock, QuantAttnBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import AttentionMap, at_loss
import torch.nn.functional as F

class new_recon_Qmodel():
    def __init__(self, args, qnn, cali_data: torch.Tensor, batch_size: int = 32, iters: int = 100, 
                act_quant: bool = False, lr_a: float = 4e-5, lr_w=1e-2, recon_w: bool = False, recon_a: bool = False):
        self.args = args
        self.model = qnn
        self.cali_data = cali_data
        self.down_name = None
        self.w_para = []
        self.a_para = []
        self.hooks = []
        self.batch_size = batch_size
        self.iters = iters
        self.act_quant = act_quant
        self.lr_a = lr_a
        self.lr_w = lr_w
        self.recon_a = recon_a
        self.recon_w = recon_w
        self.device = next(self.model.parameters()).device

    def recon_model(self, module: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in module.named_children():
            if self.down_name == None and name == 'down':
                self.down_name = 'down'
            if self.down_name == 'down' and name == '1' and isinstance(module, BaseQuantBlock) == 0:
                self.hooks.append(AttentionMap(module.block[0]))
                self.block_para(module.block[0], self.recon_w, self.recon_a)
                self.hooks.append(AttentionMap(module.attn[0]))
                self.block_para(module.attn[0], self.recon_w, self.recon_a)
                self.hooks.append(AttentionMap(module.block[1]))
                self.block_para(module.block[1], self.recon_w, self.recon_a)
                self.hooks.append(AttentionMap(module.attn[1]))
                self.block_para(module.attn[1], self.recon_w, self.recon_a)
                self.hooks.append(AttentionMap(module.downsample.conv))
                self.layer_para(module.downsample.conv, self.recon_w, self.recon_a)
                # print('reconstruction for down 1 modulelist')
                self.down_name = 'over'
            elif isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    # print('Reconstruction for layer {}'.format(name))
                    self.hooks.append(AttentionMap(module))
                    self.layer_para(module, self.recon_w, self.recon_a)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    # print('Reconstruction for block {}'.format(name))
                    self.hooks.append(AttentionMap(module))
                    self.block_para(module, self.recon_w, self.recon_a)
            elif name == 'up':# Unet的上采样过程，按顺序重建
                for up_name, up_module in reversed(list(module.named_children())):
                    if up_name == '1':
                        self.hooks.append(AttentionMap(up_module.block[0]))
                        self.block_para(up_module.block[0], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.attn[0]))
                        self.block_para(up_module.attn[0], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.block[1]))
                        self.block_para(up_module.block[1], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.attn[1]))
                        self.block_para(up_module.attn[1], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.block[2]))
                        self.block_para(up_module.block[2], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.attn[2]))
                        self.block_para(up_module.attn[2], self.recon_w, self.recon_a)
                        self.hooks.append(AttentionMap(up_module.upsample.conv))
                        self.layer_para(up_module.upsample.conv, self.recon_w, self.recon_a)
                        # print('reconstruction for up 1 modulelist')
                    elif isinstance(up_module, QuantModule):
                        if up_module.ignore_reconstruction is True:
                            print('Ignore reconstruction of layer {}'.format(up_name))
                            continue
                        else:
                            # print('Reconstruction for layer {}'.format(up_name))
                            self.hooks.append(AttentionMap(up_module))
                            self.layer_para(up_module, self.recon_w, self.recon_a)
                    elif isinstance(up_module, BaseQuantBlock):
                        if up_module.ignore_reconstruction is True:
                            print('Ignore reconstruction of block {}'.format(up_name))
                            continue
                        else:
                            # print('Reconstruction for block {}'.format(up_name))
                            self.hooks.append(AttentionMap(up_module))
                            self.block_para(up_module, self.recon_w, self.recon_a)
                    else:
                        self.recon_model(up_module)
            else:
                self.recon_model(module)

            
    def recon(self):
        # 将每一个block，module放入hooks，并将要调节参数放入w_para,a_para
        self.recon_model(self.model)
        # 获取全精度输出
        self.model.set_quant_state(weight_quant=False, act_quant=False)
        batch_size = 256
        cached_batches = []
        for i in range(int(self.cali_data[0].size(0) / batch_size)):
            out = []
            with torch.no_grad():
                model_out = self.model([_[i * batch_size : (i + 1) * batch_size] for _ in self.cali_data])
            out.append(model_out.cpu())
            for j in range(len(self.hooks)):
                out.append(self.hooks[j].out.cpu())

            cached_batches.append(out)
        torch.cuda.empty_cache()

        cached_outs = []
        for i in range(len(cached_batches[0])):
            cached_out = torch.cat([x[i] for x in cached_batches])
            cached_outs.append(cached_out)

        w_opt, a_opt = None, None
        a_scheduler, w_scheduler = None, None
        if len(self.w_para) != 0:
            w_opt = torch.optim.Adam(self.w_para, lr=self.lr_w)
            w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, T_max=self.iters, eta_min=0.)
        if len(self.a_para) != 0:
            a_opt = torch.optim.Adam(self.a_para, lr=self.lr_a)
            a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=self.iters, eta_min=0.)

        sz = cached_outs[0].size(0)
        self.model.set_quant_state(True, self.act_quant)
        for iter in range(self.iters):

            if w_opt:
                w_opt.zero_grad()
            if a_opt:
                a_opt.zero_grad()

            # 随机抽取batch个样本，获得量化输出
            idx = random.sample(range(sz), self.batch_size)
            q_out = []
            
            qmodel_out = self.model([_[idx] for _ in self.cali_data])
            q_out.append(qmodel_out)
            for j in range(len(self.hooks)):
                q_out.append(self.hooks[j].out)

            nor_loss = 1.0 * F.l1_loss(q_out[0], cached_outs[0][idx].to(self.device))*self.batch_size

            att_loss = 0.0
            att_list = []
            for i in range(1, len(q_out)):
                att_list.append(at_loss(q_out[i], cached_outs[i][idx].to(self.device)))
                att_loss = att_loss + at_loss(q_out[i], cached_outs[i][idx].to(self.device))
            att_loss = 1.0 * att_loss/(len(att_list))

            loss = nor_loss + att_loss
            loss.backward()#retain_graph=True
            if iter % 5 == 0 or iter == self.iters-1:
                print('Total loss:   {:.3f} (nor:  {:.3f}, att:  {:.3f})  block loss:   {:.3f}  iter={}'.format(
                    float(loss), float(nor_loss), float(att_loss), att_list[-3], iter))

            if w_opt:
                w_opt.step()
            if a_opt:
                a_opt.step()
            # if w_scheduler:
            #     w_scheduler.step()
            # if a_scheduler:
            #    a_scheduler.step()
        torch.cuda.empty_cache()

        for module in self.model.modules():
            if isinstance(module, QuantModule):
                '''weight '''
                if module.split == 0:
                    module.weight_quantizer.soft_targets = False
                    module.act_quantizer.is_training = False
                else:
                    module.weight_quantizer.soft_targets = False
                    module.weight_quantizer_0.soft_targets = False
                    module.act_quantizer.is_training = False
                    module.act_quantizer_0.is_training = False
            if isinstance(module, QuantAttnBlock):
                module.act_quantizer_q.is_training = False
                module.act_quantizer_k.is_training = False
                module.act_quantizer_v.is_training = False
                module.act_quantizer_w.is_training = False
        for hook in self.hooks:
            hook.remove()

        return self.model


    def block_para(self, block: BaseQuantBlock, recon_w: bool = False, recon_a: bool = False):
        round_mode = 'learned_hard_sigmoid'
        for module in block.modules():
            '''weight'''
            if isinstance(module, QuantModule):
                if module.split == 0:
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data)
                    if self.recon_w:                                        
                        module.weight_quantizer.soft_targets = True
                        self.w_para += [module.weight_quantizer.alpha]
                else :
                    module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data[:, :module.split, ...])
                    module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                                weight_tensor=module.org_weight.data[:, module.split:, ...])
                    if self.recon_w: 
                        module.weight_quantizer.soft_targets = True
                        module.weight_quantizer_0.soft_targets = True
                        self.w_para += [module.weight_quantizer.alpha]
                        self.w_para += [module.weight_quantizer_0.alpha]
            '''activation'''
            if isinstance(module, (QuantModule, BaseQuantBlock)):
                if self.act_quant and isinstance(module, QuantAttnBlock):
                    module.act_quantizer_q.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_q.delta))
                    module.act_quantizer_k.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_k.delta))
                    module.act_quantizer_v.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_v.delta))
                    module.act_quantizer_w.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_w.delta))
                    if self.recon_a:
                        self.a_para += [module.act_quantizer_q.delta]
                        self.a_para += [module.act_quantizer_k.delta]
                        self.a_para += [module.act_quantizer_v.delta]
                        self.a_para += [module.act_quantizer_w.delta]
                        module.act_quantizer_q.is_training = True
                        module.act_quantizer_k.is_training = True
                        module.act_quantizer_v.is_training = True
                        module.act_quantizer_w.is_training = True
                if self.act_quant and module.act_quantizer.delta is not None:
                    if module.split == 0:
                        module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                        if self.recon_a:
                            self.a_para += [module.act_quantizer.delta]
                            module.act_quantizer.is_training = True
                    else:
                        module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                        module.act_quantizer_0.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_0.delta))
                        if self.recon_a:
                            self.a_para += [module.act_quantizer.delta]
                            self.a_para += [module.act_quantizer_0.delta]
                            module.act_quantizer.is_training = True
                            module.act_quantizer_0.is_training = True

    def layer_para(self, layer: QuantModule, recon_w: bool = False, recon_a: bool = False):
        round_mode = 'learned_hard_sigmoid'
        layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                                weight_tensor=layer.org_weight.data)
        if self.recon_w:
            layer.weight_quantizer.soft_targets = True
            self.w_para += [layer.weight_quantizer.alpha]
        '''activation'''
        if self.act_quant and layer.act_quantizer.delta is not None:
            layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
            if self.recon_a:
                self.a_para += [layer.act_quantizer.delta]
                layer.act_quantizer.is_training = True