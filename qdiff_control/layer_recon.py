import torch
import random
# import linklink as link
from qdiff.quant_layer import QuantModule, StraightThrough, lp_loss
from qdiff.quant_model import QuantModel
from qdiff_control.block_recon import LinearTempDecay
from qdiff_control.adaptive_rounding import AdaRoundQuantizer
from qdiff_control.data_utils import save_inp_oup_data
from qdiff.utils import AttentionMap, at_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt

def layer_reconstruction(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr_a: float = 4e-5, lr_w=1e-2, p: float = 2.0,
                         input_prob: float = 1.0, keep_gpu: bool = True, 
                         recon_w: bool = False, recon_a: bool = False, add_loss: float = 0.0):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
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
    :param keep_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """

    '''set state'''                                    
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    '''set quantizer'''
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], []
    '''weight'''
    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,
                                               weight_tensor=layer.org_weight.data)
    if recon_w:
        layer.weight_quantizer.soft_targets = True
        w_para += [layer.weight_quantizer.alpha]
    '''activation'''
    if act_quant and layer.act_quantizer.delta is not None:
        layer.act_quantizer.delta = torch.nn.Parameter(torch.tensor(layer.act_quantizer.delta))
        if recon_a:
            a_para += [layer.act_quantizer.delta]
            layer.act_quantizer.is_training = True

    w_opt, a_opt = None, None
    a_scheduler, w_scheduler = None, None
    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=lr_w)
        w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_opt, T_max=iters, eta_min=0.)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr_a)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)

    loss_mode = 'none'
    rec_loss = opt_mode
    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)

    '''get input and set scale'''
    Resblock, cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, asym, act_quant, batch_size=batch_size, input_prob=True, keep_gpu=keep_gpu)

    device = 'cuda'
    sz = cached_outs.size(0)
    model.block_count = model.block_count + 1
    for i in range(iters):
        idx = random.sample(range(sz), batch_size)
        cur_out = cached_outs[idx].to(device)
        cur_inp, cur_sym = cached_inps[0][idx].to(device), cached_inps[1][idx].to(device)
        if input_prob < 1.0:
            cur_inp = torch.where(torch.rand_like(cur_inp) < input_prob, cur_inp, cur_sym)

        if w_opt:
            w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()

        out_quant = layer(cur_inp)

        loss = loss_func(out_quant, cur_out)
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

    layer.weight_quantizer.soft_targets = False
    layer.act_quantizer.is_training = False


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
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
            round_vals = self.layer.weight_quantizer.get_soft_targets()
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

