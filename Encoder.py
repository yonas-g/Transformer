import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MLP
from modules.AddNorm import AddNorm
from modules.Attention import MultiHeadSelfAttention
from utils import get_extended_attention_mask


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


class Encoder(nn.Module):
    
        def __init__(self, config):
            super(Encoder, self).__init__()
    
            self.config = config
    
            # embedding
            self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.positional_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            # self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)

            self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
            self.register_buffer('position_ids', position_ids) # from anlp

            # encoder layers
            self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])


        def embed(self, input_ids):
            """
            :param input: [batch_size, seq_len]
            """
            input_shape = input_ids.size()
            seq_length = input_shape[1]

            input_embedding = self.embedding(input)

            pos_ids = self.position_ids[:, :seq_length]
            position_embedding = self.positional_embedding(pos_ids)

            # tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            # tk_type_embeds = self.tk_type_embedding(tk_type_ids)

            embeddings = input_embedding + position_embedding #+ tk_type_embeds
            embeddings = self.dropout(embeddings)

            return embeddings
        
             
        def forward(self, input, attention_mask):
            """
            :param input: [batch_size, seq_len]
            :param attention_mask: [batch_size, seq_len]
            :return: [batch_size, seq_len, hidden_size]
            """
            attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
            # embedding
            hidden_states = self.embed(input)
    
            # encoder layers
            for encoder_layer in self.encoder_layers:
                hidden_states = encoder_layer(hidden_states, attention_mask)
    
            return hidden_states

