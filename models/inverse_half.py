import torch
from torch import nn
from .submodel import HourGlass
from .base import ResidualBlock

class InverseHalf(nn.Module):
    def __init__(self):
        super(InverseHalf, self).__init__()
        self.net = HourGlass(inChannel=1, outChannel=1)

    def forward(self, x):
        grayscale = self.net(x)
        return grayscale

class InverseHalfPRL(nn.Module):
    """
    Implementation of Xia's InvHalf model. \n
    url: https://link.springer.com/content/pdf/10.1007/978-3-030-20876-9_33.pdf \n
    ref: Xia, Menghan, and Tien-Tsin Wong. "Deep inverse halftoning via progressively residual learning." In Asian Conference on Computer Vision, pp. 523-539. Springer, Cham, 2018.

    """
    def __init__(self):
        super(InverseHalfPRL, self).__init__()
        self.net = HourGlass(inChannel=1, outChannel=1)
        self.refineNet = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            *[ResidualBlock(64) for _ in range(8)],
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        pseudo_grayscale = self.net(x)
        enhanced_detail = self.refineNet(torch.cat([x, pseudo_grayscale], dim=1))
        grayscale = pseudo_grayscale + enhanced_detail
        return grayscale, pseudo_grayscale
