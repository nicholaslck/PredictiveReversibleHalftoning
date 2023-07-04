import logging
import os
import torch
from torch.nn import functional as F
from torchvision.models.vgg import vgg19

from .smoothing import GaussianSmoothing
from .ssim import ssim
from models.inverse_half import InverseHalf
from .convert import bgr2gray


def l1_loss(input, target):
    return F.l1_loss(input, target)


def l2_loss(input, target):
    return F.mse_loss(input, target)


def gaussian_l2(input, target):
    """ data range [-1,1] """
    smoother = GaussianSmoothing(channels=1, kernel_size=11, sigma=2.0)
    return F.mse_loss(smoother(input), smoother(bgr2gray(target)))


def bin_l1(input):
    """ data range is [-1,1] """
    return (input.abs() - 1.0).abs().mean()


def ssim_loss(input, target):
    """ data range is [-1,1] """
    _ssim = ssim(input / 2. + 0.5, bgr2gray(target / 2. + 0.5), window_size=11)
    return float(1) - _ssim


class FeatureLoss:
    def __init__(self, pretrained_path, require_grad=False):
        self.featureExactor = torch.nn.parallel.DataParallel(InverseHalf()).cuda()
        logging.info("FeatureLoss: loading feature extractor: {} ...".format(pretrained_path))
        self.featureExactor.load_state_dict(torch.load(pretrained_path)['state_dict'])
        logging.info("FeatureLoss: feature network loaded")
        if not require_grad:
            for param in self.featureExactor.parameters():
                param.requires_grad = False

    def __call__(self, input, target):
        inFeature = self.featureExactor(input)
        return l2_loss(inFeature, target)


class Vgg19Loss:
    def __init__(self, disable_cuda=False):
        # data in BGR format, [0,1] range
        self.mean = [0.485, 0.456, 0.406]
        self.mean.reverse()
        self.std = [0.229, 0.224, 0.225]
        self.std.reverse()
        vgg = vgg19(pretrained=True)
        # maxpoll after conv4_4
        self.featureExactor = torch.nn.Sequential(*list(vgg.features)[:28]).eval()
        for param in self.featureExactor.parameters():
            param.requires_grad = False
        if torch.cuda.is_available() and not disable_cuda:
            self.featureExactor = torch.nn.parallel.DataParallel(self.featureExactor).cuda()
        # print('[*] Vgg19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def __call__(self, input, target):
        inFeature = self.featureExactor(self.normalize(input).flip(1))
        targetFeature = self.featureExactor(self.normalize(target).flip(1))
        return l2_loss(inFeature, targetFeature)
