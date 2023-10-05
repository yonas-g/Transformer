import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()

        self.dense_expansion = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_contraction = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, input):
        """
        :param input: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, intermediate_size]
        """
        
        output = self.dense_expansion(input)
        output = self.activation(output)
        output = self.dense_contraction(output)
        return output