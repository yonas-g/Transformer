# Transformers

Minimal PyTorch implementation of the Transformers architecture from Vaswani et al.'s 2017 paper, [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This repository serves as a deep dive into understanding key architectures and their nuances, including architecture design and training techniques.

## Components
- Tokenizer (Embedding)

- Modules:
    - Attention
        - MultiHeadAttentionBase
        - MultiHeadSelfAttention
        - MultiHeadCrossAttention
    - AddNorm
    - MLP

- Encoder:
    - EncoderLayer
    - Encoder

- Decoder
    - DecoderLayer
    - Decoder
    