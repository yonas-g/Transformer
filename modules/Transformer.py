import torch
import torch.nn as nn

from modules.Encoder import Encoder
from modules.Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, encoder_input, decoder_input, encoder_attention_mask, decoder_attention_mask):
        """
        :param encoder_input: [batch_size, src_len]
        :param decoder_input: [batch_size, tgt_len]
        :param encoder_attention_mask: [batch_size, 1, 1, src_len]
        :param decoder_attention_mask: [batch_size, 1, tgt_len, tgt_len]
        """

        encoder_output = self.encoder(encoder_input, encoder_attention_mask)
        decoder_output = self.decoder(decoder_input, encoder_output, encoder_attention_mask, decoder_attention_mask)
        
        return decoder_output