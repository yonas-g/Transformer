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
        
        # embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)

    
    def embed(self, input_ids):
        """
        :param input_ids: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        input_embed = self.embedding(input_ids)
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]
        position_embed = self.positional_embedding(position_ids)
        # embed
        embed = self.embed_layer_norm(input_embed + position_embed)
        embed = self.dropout(embed)

        return embed

    
    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """
        :param tgt: [batch_size, tgt_len]
        :param tgt_mask: [batch_size, 1, tgt_len, tgt_len]
        :param memory: [batch_size, src_len, hidden_size]
        :param memory_mask: [batch_size, 1, 1, src_len]
        :return: [batch_size, tgt_len, hidden_size]
        """
        # embed
        embed = self.embed(tgt)
        # decoder layers
        for layer in self.layers:
            embed = layer(embed, tgt_mask, memory, memory_mask)
        
        # output layer
        output = self.output_layer(embed)
        output = F.softmax(output, dim=-1)
        
        return output