# Transformers

Minimal PyTorch implementation of the Transformers architecture from Vaswani et al.'s 2017 paper, [Attention Is All You Need](https://arxiv.org/abs/1706.03762). This repository serves as a deep dive into understanding key architectures and their nuances, including architecture design and training techniques.

## Components


## Project Structure

- `Tokenizer.py`
- `Modules/`
    - `Attention.py`
        - `MultiHeadAttentionBase`
        - `MultiHeadSelfAttention`
        - `MultiHeadCrossAttention`
    - `AddNorm.py`
    - `MLP.py`
    - `Encoder.py`
        - `EncoderLayer`
        - `Encoder`
    - `Decoder.py`
        - `DecoderLayer`
        - `Decoder`
    - `Transformer.py`
        - `Transformer`
- `Config/`
    - `Config.py`
-  `BERT/`
    - `BERT.py`
        - `Bert`
    