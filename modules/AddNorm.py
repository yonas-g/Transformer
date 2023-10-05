import torch
import torch.nn as nn


class AddNorm(nn.Module):

    def __init__(self, config):
        super(AddNorm, self).__init__()

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self):
        pass