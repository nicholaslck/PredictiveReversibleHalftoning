import torch

def psnr(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal to Noise Ratio
    input and target have range [0, 255]"""

    mse = torch.mean((input - target) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))

class PSNR(torch.nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, input, target):
        return psnr(input, target)
