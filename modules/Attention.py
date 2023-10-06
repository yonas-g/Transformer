import math

import torch
import torch.nn as nn


class MultiHeadAttentionBase(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttentionBase, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform_layers(self, x, layer):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :param layer: nn.Linear
        :return: [batch_size, num_attention_heads, seq_len, attention_head_size]
        """
        B, seq_len = x.shape[:2]
        x = layer(x)
        x = x.view(B, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)

    def attention(self, key, query, value, attention_mask=None):
        """
        :param key: [batch_size, num_attention_heads, seq_len, attention_head_size]
        :param query: [batch_size, num_attention_heads, seq_len, attention_head_size]
        :param value: [batch_size, num_attention_heads, seq_len, attention_head_size]
        :param attention_mask: [batch_size, 1, 1, seq_len]
        """
        score = torch.matmul(query, key.transpose(-1, -2))

        if attention_mask is not None:
            score = score +  attention_mask # # -10000.0 -> 0 (softmax(-10000.0) -> 0)

        score = score / math.sqrt(self.attention_head_size)

        score = torch.softmax(score, dim=-1)

        score = self.dropout(score)

        attn = torch.matmul(score, value) # [B, num_attention_heads, seq_len, attention_head_size]

        attn = attn.permute(0, 2, 1, 3).contiguous() # [B, seq_len, num_attention_heads, attention_head_size]
        B, seq_len = attn.size()[:2]

        return attn.view((B, seq_len, self.all_head_size)) # or attn.view((B, seq_len, -1)) # [B, seq_len, hidden_size]



class MultiHeadSelfAttention(MultiHeadAttentionBase):
    def forward(self, x, attention_mask=None):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """

        key = self.transform_layers(x, self.key)
        value = self.transform_layers(x, self.value)
        query = self.transform_layers(x, self.query)

        attn = self.attention(key, query, value, attention_mask)

        return attn


class MultiHeadCrossAttention(MultiHeadAttentionBase):
    def forward(self, x, memory, memory_mask=None):
        """
        :param x: [batch_size, seq_len, hidden_size]
        :param memory: [batch_size, seq_len, hidden_size]
        :param memory_mask: [batch_size, 1, 1, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """

        key = self.transform_layers(memory, self.key)
        value = self.transform_layers(memory, self.value)
        query = self.transform_layers(x, self.query)

        attn = self.attention(key, query, value, memory_mask)

        return attn