import torch
import torch.nn as nn


class AddNorm(nn.Module):
    """
    Add and Norm
    Applied after Multi-Head Attention and Feed Forward Network
    """
    def __init__(self, config):
        super(AddNorm, self).__init__()

        self.layer_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input, output):
        """
        :param input: [batch_size, seq_len, hidden_size]
        :param output: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, hidden_size]
        """
        output = self.layer_dense(output)
        output = self.dropout(output)
        output = self.layer_norm(input + output)
        return output