import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP
from modules.AddNorm import AddNorm
from modules.Attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):

    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.config = config

        # Multi-Head Attention
        self.self_attention = MultiHeadSelfAttention(config)
        # add and norm
        self.add_norm_1 = AddNorm(config)
        # feed forward
        self.feed_forward = MLP(config)
        # add and norm
        self.add_norm_2 = AddNorm(config)
        
    def forward(self, input, attention_mask):
        """
        :param input: [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        # self-attention
        self_attention_output = self.self_attention(input, attention_mask)
        # add and norm
        add_norm_1_output = self.add_norm_1(input, self_attention_output)
        # feed forward
        feed_forward_output = self.feed_forward(add_norm_1_output)
        # add and norm
        add_norm_2_output = self.add_norm_2(add_norm_1_output, feed_forward_output)

        return add_norm_2_output

