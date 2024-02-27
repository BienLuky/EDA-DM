import numpy as np
import torch
import random
from qdiff.quant_layer import QuantModule, lp_loss
from qdiff.quant_model import QuantModel
from qdiff.quant_block import BaseQuantBlock, QuantAttnBlock, QuantQKMatMul, QuantSMVMatMul, QuantAttentionBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.data_utils import save_inp_oup_data
from qdiff.utils import AttentionMap, at_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt

def block_reconstruction(model: QuantModel, block: BaseQuantBlock, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr_a: float = 4e-5, lr_w=1e-2, p: float = 2.0,
                         input_prob: float = 1.0, keep_gpu: bool = True, 
                         recon_w: bool = False, recon_a: bool = False, add_loss: float = 0.0, change_block: bool = False):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    """

    '''set state'''                                    
    # model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'
    hooks = []
    '''set quantizer'''
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    for module in block.modules():
        '''weight'''
        if isinstance(module, QuantModule):
            hooks.append(AttentionMap(module))
            if module.split == 0:
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data)
                if recon_w:                                        
                    module.weight_quantizer.soft_targets = True
                    w_para += [module.weight_quantizer.alpha]
            else :
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data[:, :module.split, ...])
                module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data[:, module.split:, ...])
                if recon_w: 
                    module.weight_quantizer.soft_targets = True
                    module.weight_quantizer_0.soft_targets = True
                    w_para += [module.weight_quantizer.alpha]
                    w_para += [module.weight_quantizer_0.alpha]
        '''activation'''
        if isinstance(module, (QuantModule, BaseQuantBlock)):
            if act_quant and isinstance(module, QuantAttentionBlock):
                module.attention.qkv_matmul.act_quantizer_q.delta = torch.nn.Parameter(torch.tensor(module.attention.qkv_matmul.act_quantizer_q.delta))
                module.attention.qkv_matmul.act_quantizer_k.delta = torch.nn.Parameter(torch.tensor(module.attention.qkv_matmul.act_quantizer_k.delta))
                module.attention.smv_matmul.act_quantizer_v.delta = torch.nn.Parameter(torch.tensor(module.attention.smv_matmul.act_quantizer_v.delta))
                module.attention.smv_matmul.act_quantizer_w.delta = torch.nn.Parameter(torch.tensor(module.attention.smv_matmul.act_quantizer_w.delta))
                if recon_a:
                    a_para += [module.attention.qkv_matmul.act_quantizer_q.delta]
                    a_para += [module.attention.qkv_matmul.act_quantizer_k.delta]
                    a_para += [module.attention.smv_matmul.act_quantizer_v.delta]
                    a_para += [module.attention.smv_matmul.act_quantizer_w.delta]
                    module.attention.qkv_matmul.act_quantizer_q.is_training = True
                    module.attention.qkv_matmul.act_quantizer_k.is_training = True
                    module.attention.smv_matmul.act_quantizer_v.is_training = True
                    module.attention.smv_matmul.act_quantizer_w.is_training = True
            if act_quant and isinstance(module, QuantAttnBlock):
                module.act_quantizer_q.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_q.delta))
                module.act_quantizer_k.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_k.delta))
                module.act_quantizer_v.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_v.delta))
                module.act_quantizer_w.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_w.delta))
                if recon_a:
                    a_para += [module.act_quantizer_q.delta]
                    a_para += [module.act_quantizer_k.delta]
                    a_para += [module.act_quantizer_v.delta]
                    a_para += [module.act_quantizer_w.delta]
                    module.act_quantizer_q.is_training = True
                    module.act_quantizer_k.is_training = True
                    module.act_quantizer_v.is_training = True
                    module.act_quantizer_w.is_training = True
            if act_quant and module.act_quantizer.delta is not None:
                if module.split == 0:
                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                    if recon_a:
                        a_para += [module.act_quantizer.delta]
                        module.act_quantizer.is_training = True
                else:
                    module.act_quantizer.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer.delta))
                    module.act_quantizer_0.delta = torch.nn.Parameter(torch.tensor(module.act_quantizer_0.delta))
                    if recon_a:
                        a_para += [module.act_quantizer.delta]
                        a_para += [module.act_quantizer_0.delta]
                        module.act_quantizer.is_training = True
                        module.act_quantizer_0.is_training = True
        

    w_opt, a_opt = None, None
    a_scheduler, w_scheduler = None, None
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=lr_w)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, T_max=iters, eta_min=0.)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr_a)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)

    # loss_mode = 'relaxation'
    loss_mode = 'none'
    rec_loss = opt_mode
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)


    '''get input and set scale'''
    Resblock, cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, act_quant, batch_size=32, input_prob=True, keep_gpu=keep_gpu)

    # if opt_mode != 'mse':
    #     cached_grads = save_grad_data(model, block, cali_data, act_quant, batch_size=batch_size)
    # else:
    #     cached_grads = None

    device = 'cuda'
    # sz = cached_inps.size(0)
    sz = cached_outs.size(0)
    model.block_count = model.block_count + 1
    module_loss_list = []
    out_loss_list = []
    # batch_size = 256
    for i in range(iters):
        idx = random.sample(range(sz), batch_size)
        # cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        # cur_grad = cached_grads[idx] if opt_mode != 'mse' else None
        if Resblock:
            cur_inp, cur_sym = cached_inps[0][0][idx].to(device), cached_inps[1][0][idx].to(device)
            temb_cur_inp, temb_cur_sym = cached_inps[0][1][idx].to(device), cached_inps[1][1][idx].to(device)
        else:
            cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx].to(device)
        if input_prob < 1.0:
            rand = torch.rand_like(cur_inp)
            cur_inp = torch.where(rand < input_prob, cur_inp, cur_sym)
        else:
            cur_inp = cur_sym

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()

        if Resblock:
            out_quant = block(cur_inp, temb_cur_inp)
        else:
            out_quant = block(cur_inp)
        
        m_loss = 0.0
        if len(hooks) != 0:
            if Resblock:
                # module full
                block.set_quant_state(False, False)
                r_out = block(cur_sym, temb_cur_sym)
                module_r = []
                for j in range(len(hooks)):
                    module_r.append(hooks[j].out)
                # module q
                block.set_quant_state(True, act_quant)
                q_out = block(cur_inp, temb_cur_inp)
                module_q = []
                for j in range(len(hooks)):
                    module_q.append(hooks[j].out)
            else:
                # module full
                block.set_quant_state(False, False)
                r_out = block(cur_sym)
                module_r = []
                for j in range(len(hooks)):
                    module_r.append(hooks[j].out)
                # module quantized
                block.set_quant_state(True, act_quant)
                q_out = block(cur_inp)
                module_q = []
                for j in range(len(hooks)):
                    module_q.append(hooks[j].out)
            
            module_loss_list.append(lp_loss(module_q[0], module_r[0], p=2).cpu().detach().numpy())
            m_loss_list = []
            for j in range(len(module_r)-1):
                loss_module = lp_loss(module_q[j], module_r[j], p=2)
                m_loss = m_loss + loss_module
                m_loss_list.append(loss_module)

        block_loss = loss_func(out_quant, cur_out)
        out_loss_list.append(block_loss.cpu().detach().numpy())
        loss = block_loss + add_loss * m_loss

        loss.backward()#retain_graph=True

        if w_opt:
            w_opt.step()
        if a_opt:
            a_opt.step()
        if w_scheduler:
            w_scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()

    for module in block.modules():
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
        if isinstance(module, QuantAttentionBlock):
            module.attention.qkv_matmul.act_quantizer_q.is_training = False
            module.attention.qkv_matmul.act_quantizer_k.is_training = False
            module.attention.smv_matmul.act_quantizer_v.is_training = False
            module.attention.smv_matmul.act_quantizer_w.is_training = False
    for hook in hooks:
        hook.remove()


class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.iters = max_count

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        # if self.count%1000 == 0 or self.count == 1:
        # # if self.count == self.iters or self.count == 1:
        #     print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #           float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        # print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #         float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            # return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
