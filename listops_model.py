from pickletools import pydict
import torch
import torch.nn.functional as F
import fftconv

import importlib

importlib.reload(fftconv)

class ModelConfig:

    scaling = 0.01
    real_init = -0.01
    bidirectional = False

    attention = False

    d_state = 64
    d_embed = 256

    num_blocks = 4

    max_len = 2048

    num_classes = 10



    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)


class ResidualStateSpaceBlock(torch.nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ss = fftconv.SimpleState(d_model=config.d_embed,
                                      d_state=config.d_state,
                                      scaling=config.scaling,
                                      real_init=config.real_init,
                                      bidirectional=config.bidirectional)

        self.mha = torch.nn.MultiheadAttention(config.d_embed,4)
        self.config = config
        self.norm = torch.nn.LayerNorm(config.d_embed)
        self.fc = torch.nn.Linear(config.d_embed, config.d_embed)

    def forward(self, x):

        if self.config.attention:
            ss_output, _ = self.mha(x, x, x)
        else:
            ss_output, final_state = self.ss(x)
        x = self.norm(x + F.gelu(ss_output))
        F.gelu(self.fc(x))


        return x



class ListOpsModel(torch.nn.Module):

    def __init__(self, config, vocab):
        super().__init__()

        self.config = config

        self.tok_embedding = torch.nn.Embedding(len(vocab), config.d_embed)
        self.pos_embedding = torch.nn.Parameter(torch.randn(config.max_len, config.d_embed)) # do we need this? idk

        self.blocks = [ResidualStateSpaceBlock(config) for _ in range(config.num_blocks)]

        self.trunk = torch.nn.Sequential(*self.blocks)

        self.head = torch.nn.Linear(config.d_embed, config.num_classes)

    def forward(self, input, lengths):

        # technically we should do some error checking to make sure that input is never longer than max_length.
        # but we won't do that right now...


        x = self.tok_embedding(input) # [B, L, H]

        B, L, H = x.size()

        x += self.pos_embedding[:L, :] # [B, L, H]

        x = self.trunk(x) # [B, L, H]


        # we only care about the prediction at the end of the sequence, which is at a different
        # point for every example:
        final_feature = torch.take_along_dim(x, indices=lengths.reshape(-1, 1, 1) - 1, dim=1) # [B, 1, H]

        logits = self.head(final_feature.squeeze()) # [B, C] (C = num classes)

        return logits









