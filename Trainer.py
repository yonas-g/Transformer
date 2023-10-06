# transfomrer model trainer
# aim: train the original transformer model

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.Encoder import Encoder
from modules.Decoder import Decoder


def train_step(model, batch, device, optimizer):
    model.train()
    src, tgt, src_mask, tgt_mask = batch
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    optimizer.zero_grad()

    output = model(src, tgt, src_mask, tgt_mask)

    loss = F.cross_entropy(output.view(-1, output.size(-1)), tgt.view(-1), ignore_index=0)

    loss.backward()

    optimizer.step()

    return loss.item()