'''This code contains the main module of the paraphrase generator.

'''
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .highway import Highway
from .encoder import Encoder
from .decoder import Decoder


class Paraphraser(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
