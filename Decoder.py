import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP
from modules.AddNorm import AddNorm
from modules.Attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from utils import get_extended_attention_mask


class DecoderLayer(nn.Module):

    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadSelfAttention(config)
        self.self_attention_norm = AddNorm(config)

        self.cross_attention = MultiHeadCrossAttention(config)
        self.cross_attention_norm = AddNorm(config)

        self.ffn = MLP(config)
        self.ffn_norm = AddNorm(config)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """
        :param tgt: [batch_size, tgt_len, hidden_size]
        :param tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        :param memory: [batch_size, src_len, hidden_size] # From the encoder
        :param memory_mask: [batch_size, 1, 1, src_len]
        :return: [batch_size, tgt_len, hidden_size]
        """

        # self attention
        self_attention_output = self.self_attention(tgt, tgt_mask)
        self_attention_output = self.self_attention_norm(tgt, self_attention_output)

        # cross attention
        cross_attention_output = self.cross_attention(self_attention_output, memory, memory_mask)
        cross_attention_output = self.cross_attention_norm(self_attention_output, cross_attention_output)

        # ffn
        ffn_output = self.ffn(cross_attention_output)
        ffn_output = self.ffn_norm(cross_attention_output, ffn_output)

        return ffn_output


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
            
    
    def forward(self, tgt, tgt_mask, memory, memory_mask):
        pass