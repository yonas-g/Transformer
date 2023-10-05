import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Attention import MultiHeadSelfAttention
from modules.AddNorm import AddNorm


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.config = config

        # Multi-Head Attention
        self.self_attention = MultiHeadSelfAttention(config)
        # add and norm
        self.add_norm_1 = AddNorm(config)
        # feed forward
        # add and norm

