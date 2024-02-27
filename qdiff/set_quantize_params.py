import torch
from .quant_layer import QuantModule
from .quant_block import BaseQuantBlock, QuantAttnBlock
from .quant_model import QuantModel
from typing import Union


def set_act_quantize_params(
    module: Union[QuantModel, QuantModule, BaseQuantBlock],
    cali_data,
    batch_size: int = 256,
):
    print(f"set_act_quantize_params")
    module.set_quant_state(True, True)

    for m in module.modules():
        # if isinstance(m, (QuantModule, BaseQuantBlock, QuantAttnBlock)):
        if isinstance(m, QuantModule):
            if m.split == 0:
                m.act_quantizer.set_inited(False)
            else:
                m.act_quantizer.set_inited(False)
                m.act_quantizer_0.set_inited(False)
        if isinstance(m, QuantAttnBlock):
            m.act_quantizer_k.set_inited(False)
            m.act_quantizer_q.set_inited(False)
            m.act_quantizer_v.set_inited(False)
            m.act_quantizer_w.set_inited(False)
    """set or init step size and zero point in the activation quantizer"""
    if not isinstance(cali_data, (tuple, list)):
        batch_size = min(batch_size, cali_data.size(0))
        with torch.no_grad():
            for i in range(int(cali_data.size(0) / batch_size)):
                module(cali_data[i * batch_size : (i + 1) * batch_size].cuda())
        torch.cuda.empty_cache()

        for t in module.modules():
            if isinstance(t, (QuantModule, BaseQuantBlock)):
                t.act_quantizer.set_inited(True)
    else:
        batch_size = min(batch_size, cali_data[0].size(0))
        with torch.no_grad():
            for i in range(int(cali_data[0].size(0) / batch_size)):
            # for i in range(batch_size / batch_size):
                module(
                    *[
                        _[i * batch_size : (i + 1) * batch_size].cuda()
                        for _ in cali_data
                    ]
                )
        torch.cuda.empty_cache()

    for m in module.modules():
        # if isinstance(m, (QuantModule, BaseQuantBlock, QuantAttnBlock)):
        if isinstance(m, QuantModule):
            if m.split == 0:
                m.act_quantizer.set_inited(True)
            else:
                m.act_quantizer.set_inited(True)
                m.act_quantizer_0.set_inited(True)
        if isinstance(m, QuantAttnBlock):
            m.act_quantizer_k.set_inited(True)
            m.act_quantizer_q.set_inited(True)
            m.act_quantizer_v.set_inited(True)
            m.act_quantizer_w.set_inited(True)

def set_weight_quantize_params(model, cali_data):
    print(f"set_weight_quantize_params")

    model.set_quant_state(True, False)

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.set_inited(False)

    batch_size = 32
    with torch.no_grad():

        model(
            *[
                _[:batch_size].cuda()
                for _ in cali_data
            ]
        )
    torch.cuda.empty_cache()

    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            if module.split == 0:
                module.weight_quantizer.set_inited(True)
            else:
                module.weight_quantizer.set_inited(True)
                module.weight_quantizer_0.set_inited(True)