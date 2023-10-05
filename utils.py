
from torch import Tensor


def get_extended_attention_mask(attention_mask: Tensor, dtype) -> Tensor: # from anlp
  # attention_mask [batch_size, seq_length]
  assert attention_mask.dim() == 2
  # [batch_size, 1, 1, seq_length] for multi-head attention
  extended_attention_mask = attention_mask[:, None, None, :]
  extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
  extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
  return extended_attention_mask