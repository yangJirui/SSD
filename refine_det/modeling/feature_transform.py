import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def add_trans_layer(in_cs, out_c):
    trans_layer = []

    for in_c in range(in_cs):
        trans_layer.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1,
                          padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=1, padding=1)
            ))
    return trans_layer


def add_upsample_layer(in_cs, paddings=[1, 1, 1, 1], out_c=256):
    '''
    use deconv to upsample
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

    .. math::
          H_out = (Hin -1)*stride - 2*padding + dila*(k-1) + output_padding + 1
          10 = (5-1)*2 -2*1 + 3 + 1
          20 = (10-1)*2 -2*1 + 3 + 1
    '''
    # for 320x320. shape 40x40, 20x20, 10x10, 5x5
    upsample_layers = []
    for in_c, p in zip(in_cs, paddings):
        upsample_layers.append(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c,
                               kernel_size=4, stride=2,
                               padding=p, output_padding=0)
        )

    return upsample_layers


def add_fuse_layer(in_c, out_c, num):
    fuse_layer = []
    for _ in range(num):
        fuse_layer.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1,
                          padding=1),
                nn.ReLU(inplace=True))
        )

    return fuse_layer


class RefineDetTransformer(nn.Module):
    def __init__(self, input_channels, out_channel=256):  # to suit py2 and py3
        """Compose a RefineDet model using the given components.
        """
        super(RefineDetTransformer, self).__init__()
        self.input_channels = input_channels
        self.out_channel = out_channel
        self.trans_layers = nn.ModuleList(add_trans_layer(in_cs=input_channels,
                                                          out_c=out_channel))
        self.upsample_layers = nn.ModuleList(add_upsample_layer(in_cs=input_channels,
                                                                out_c=out_channel))

        self.fuse_layers = nn.ModuleList(add_fuse_layer(in_c=out_channel,
                                                        out_c=out_channel,
                                                        num=len(input_channels)))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        assert isinstance(x, list), "the input of refineDetTransformer must be a list"

        

