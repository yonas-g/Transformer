import torch
import torch.nn as nn
from collections import OrderedDict

from ..config import Config
from ..modules.Encoder import EncoderLayer


class Bert(nn.Module):
      def __init__(self, config_dict):
        super(Bert, self).__init__()
        
        self.config = Config.from_dict(config_dict)

        self.embeddings = nn.ModuleDict({
          'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
          'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
          'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        self.ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            EncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))


      def forward(self, input_ids, attention_mask=None, token_type_ids=None, ):
        """
        Input:
            - input_ids: tokenized input: [B, seq_len]. seq_len is padded to max_len of 512. So, [B, 512]
            - attention_mask: [B, seq_len]
            - token_type_ids: [B, seq_len]
        all inputs are padded to max_length

        returns:
            - x: [B, seq_len, hidden_size]
            - o: [B, hidden_size]
        """
        
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        ## sum instead of cat for the embeddings
        x = self.embeddings.token(input_ids) + \
            self.embeddings.position(position_ids) + \
            self.embeddings.token_type(token_type_ids) # x: [B, seq_len, hidden_size]

        x = self.dropout(self.ln(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return (x, o)
      

      def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self