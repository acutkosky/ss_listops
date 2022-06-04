from pickletools import pydict
import torch
import torch.nn.functional as F
import fftconv

class ListOpsModelConfig:

    scaling = 0.1
    real_init = -0.5
    bidirectional = False

    d_state = 64
    d_embed = 64

    num_blocks = 4

    max_len = 2048

    num_classes = 10



    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k,v)


class ResidualStateSpaceBlock(torch.nn.Module):

    def __init__(self, config):
        super().__init__(self)

        self.ss = fftconv.SimpleState(d_embed=config.d_embed,
                                      d_state=config.d_state,
                                      scaling=config.scaling,
                                      real_init=config.real_init,
                                      bidirectional=config.bidirectional)

        self.norm = torch.nn.LayerNorm(config.d_embed)

    def forward(self, x):

        x = x + self.ss(x)
        x = self.norm(x)

        return x



class ListOpsModel(torch.nn.Module):

    def __init__(self, config, vocab):
        super().__init__(self)

        self.config = config

        self.tok_embedding = torch.nn.Embedding(len(vocab), config.d_embed)
        self.pos_embedding = torch.Parameter(torch.randn(config.max_len, config.d_embed)) # do we need this? idk

        self.blocks = [ResidualStateSpaceBlock(config) for _ in range(config.num_blocks)]

        self.trunk = torch.nn.Sequential(*self.blocks)

        self.head = torch.nn.Linear(config.d_embed, config.num_classes)

    def forward(self, input, lengths):

        # technically we should do some error checking to make sure that input is never longer than max_length.
        # but we won't do that right now...

        x = self.tok_embedding(input) # [B, L, H]
        x += self.pos_embedding # [B, L, H]

        x = self.trunk(x) # [B, L, H]

        # we only care about the prediction at the end of the sequence, which is at a different
        # point for every example:
        final_feature = torch.take_along_dim(x, indices=lengths.reshape(-1, 1, 1), dim=1) # [B, 1, H]

        logits = self.head(final_feature.squeeze()) # [B, C] (C = num classes)

        return logits

        






        





    
    def forward(self):


