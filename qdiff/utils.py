import logging
import torch
import torch.nn as nn
from qdiff.quant_layer import QuantModule
from qdiff.quant_block import BaseQuantBlock
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        # self.ac_output = []

    def hook_fn(self, module, input, output):
        self.out = output
        self.feature = input
        # self.ac_output.append(output)

    def remove(self):
        self.hook.remove()

        
def at(x):
    return x.view(x.size(0), -1)


def at_loss(x, y):
    batch_mean = (at(x) - at(y)).pow(2).mean(1)
    return batch_mean.sum()


def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
