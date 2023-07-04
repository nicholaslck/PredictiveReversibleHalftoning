import logging
from logging import info as log
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Function
# from .base import ResidualBlock, SkipConnection, ConvBlock

from .inverse_half import InverseHalf, InverseHalfPRL

from .submodel import HourGlass
from utils.dct import DCT_Lowfrequency
from utils.convert import *


class ResHalfPredictor(nn.Module):
    def __init__(
            self, train=False, encoder_pretrained=None, invhalf_pretrained=None, stage=2, use_input_y=False,
            noise_weight=0.3):
        super(ResHalfPredictor, self).__init__()
        self.is_train = train
        self.use_input_y = use_input_y
        self.noise_weight = noise_weight
        self.stage = stage

        if encoder_pretrained:
            self.load_pretrained_encoder(encoder_pretrained)
        else:
            self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        self.decoder = HourGlass(inChannel=1, outChannel=2, resNum=4, convNum=4)
        self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.

        if invhalf_pretrained:
            self.load_pretrained_invhalf(invhalf_pretrained)
        else:
            self.invhalf = InverseHalfPRL()

        if train:
            if stage == 1:
                self._froze_module(self.invhalf) 
            elif stage == 2:
                self._froze_module(self.encoder) # type: ignore
                self._froze_module(self.decoder) 
            else:
                raise ValueError(f"Invalid value for stage {stage}.")

    def _froze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def load_pretrained_encoder(self, checkpoint_path):
        __reshalf_model_pll = nn.parallel.DataParallel(ResHalf())
        __reshalf_model_pll.load_state_dict(torch.load(checkpoint_path)["state_dict"], strict=False)
        self.encoder: HourGlass = __reshalf_model_pll.module.encoder # type: ignore
        log("ResHalf pretrained encoder loaded.")
    
    def load_pretrained_invhalf(self, checkpoint_path):
        __invhalf_model_pll = nn.parallel.DataParallel(InverseHalfPRL())
        __invhalf_model_pll.load_state_dict(torch.load(checkpoint_path)["model_state_dict"], strict=True)
        invhalf_model: InverseHalfPRL = __invhalf_model_pll.module  # type: ignore
        self.invhalf = invhalf_model
        log("InverseHalfPRL pretrained model loaded.")

    def load_pretrained_invhalf_from_reshalfpredictor(self, checkpoint_path, device=None):
        __reshalfpred_model_pll = torch.nn.parallel.DataParallel(ResHalfPredictor()).to(device)
        __reshalfpred_model_pll.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"], strict=True)
        self.invhalf: InverseHalfPRL = __reshalfpred_model_pll.module.invhalf # type: ignore
        log("InverseHalfPRL pretrained model loaded from another ResHalfPredictor module.")

    def load_checkpoint_from_path(self, path: str, strict=True):
        state_dict = torch.load(path)['model_state_dict']
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        log(f"missing_keys: {missing_keys}")
        log(f"unexpected_keys: {unexpected_keys}")

    def forward(self, *x):
        # x[0]: color_image
        # x[1]: ref_halftone - ov halftone
        noise_size = (x[0].size()[0], 1, x[0].size()[2], x[0].size()[3])
        noise = self.noise_weight * torch.randn(noise_size, dtype=x[0].dtype, device=x[0].device)
        halftone = self.encoder(torch.cat((x[0], noise), dim=1))
        halftone_q = self.quantizer(halftone)
        crcb = self.decoder(halftone_q)

        pseudo_y = None
        if self.use_input_y:
            y = bgr2gray(x[0])
            pseudo_y = y
        else:
            # y, pseudo_y = self.invhalf(halftone_q)
            y, pseudo_y = self.invhalf(halftone)

        restored = torch.cat((y, crcb), dim=1)

        # convert YCrCb to BGR
        restored = range_cvt(restored, RANGE_MODE_NEGATIVE_1_TO_1, RANGE_MODE_0_TO_1)
        restored = ycrcb2bgr(restored)
        restored = range_cvt(restored, RANGE_MODE_0_TO_1, RANGE_MODE_NEGATIVE_1_TO_1)

        if self.is_train:
            if self.stage == 1:
                halfDCT = self.dcter(halftone / 2. + 0.5)
                refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
                return halftone, restored, halfDCT, refDCT, halftone_q, y, crcb, pseudo_y
            else:
                return halftone, restored, None, None, halftone_q, y, crcb, pseudo_y
        else:
            return halftone, restored, halftone_q

    def decode(self, x):
        halftone_q = self.quantizer(x)
        crcb = self.decoder(halftone_q)
        pseudo_y = None
        if self.use_input_y:
            y = bgr2gray(x)
            pseudo_y = y
        else:
            y, pseudo_y = self.invhalf(halftone_q)
        restored = torch.cat((y, crcb), dim=1)

        # convert YCrCb to BGR
        restored = range_cvt(restored, RANGE_MODE_NEGATIVE_1_TO_1, RANGE_MODE_0_TO_1)
        restored = ycrcb2bgr(restored)
        restored = range_cvt(restored, RANGE_MODE_0_TO_1, RANGE_MODE_NEGATIVE_1_TO_1)
        return restored, y, crcb, pseudo_y


class ResHalf_Luminance(nn.Module):
    def __init__(self, train=True, warm_stage=False, encoder_pretrained=None, invhalf_pretrained=None, option=None):
        super(ResHalf_Luminance, self).__init__()
        self.isTrain = train
        self.option = option if option is not None else {}

        self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        self.decoder = HourGlass(inChannel=1, outChannel=2, resNum=4, convNum=4)
        self.dcter = DCT_Lowfrequency(size=256, fLimit=self.option.get("fLimit", 50))
        # self.dcter = DCT_Lowfrequency(size=512, fLimit=self.option.get("fLimit", 50))
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.invhalf = InverseHalf()

        if encoder_pretrained:
            __reshalf_model_pll = nn.parallel.DataParallel(ResHalf())
            __dict = torch.load(encoder_pretrained)["state_dict"]
            __reshalf_model_pll.load_state_dict(__dict, strict=False)
            reshalf_model: ResHalf = __reshalf_model_pll.module  # type: ignore
            self.encoder = reshalf_model.encoder
            logging.debug("ResHalf pretrained encoder loaded.")

        if invhalf_pretrained:
            __invhalf_model_pll = nn.parallel.DataParallel(InverseHalf())
            __dict = torch.load(invhalf_pretrained)["state_dict"]
            __invhalf_model_pll.load_state_dict(__dict, strict=False)
            invhalf_model: InverseHalf = __invhalf_model_pll.module  # type: ignore
            self.invhalf = invhalf_model
            logging.debug("InverseHalf pretrained model loaded.")

        if warm_stage:
            for _, param in self.decoder.named_parameters():
                param.requires_grad = False

        if self.option.get("invhalf_freeze", False):
            for _, param in self.invhalf.named_parameters():
                param.requires_grad = False

    def forward(self, *x):
        # x[0]: color_image
        # x[1]: ref_halftone
        # x[2]: gray_image
        # noise = torch.randn_like(x[1]) * 0.3
        noise = torch.randn((x[0].size()[0], 1, x[0].size()[2], x[0].size()[3]),
                            dtype=x[0].dtype, device=x[0].device) * 0.3
        halfRes = self.encoder(torch.cat((x[0], noise), dim=1)) # type: ignore
        halfResQ = self.quantizer(halfRes)
        if self.option["invhalf_enabled"]:
            l_restore = self.invhalf(halfResQ)
        else:
            l_restore = x[2]
        ab_restore = self.decoder(halfResQ)
        restored = torch.cat((l_restore, ab_restore), dim=1)

        restored = range_cvt(restored, RANGE_MODE_NEGATIVE_1_TO_1, RANGE_MODE_0_TO_1)
        if self.option.get("channel_mode", None) == "lab":
            restored = lab2bgr(restored)
        else:
            restored = ycrcb2bgr(restored)
        restored = range_cvt(restored, RANGE_MODE_0_TO_1, RANGE_MODE_NEGATIVE_1_TO_1)

        if self.isTrain:
            halfDCT = self.dcter(halfRes / 2. + 0.5)
            refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
            return halfRes, halfDCT, refDCT, restored, halfResQ, l_restore, ab_restore
        else:
            return halfRes, restored


class ResHalf(nn.Module):
    def __init__(self, train=True, warm_stage=False, large_decoder=False):
        super(ResHalf, self).__init__()
        self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        if not large_decoder:
            self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)
        else:
            # self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=8, convNum=8)
            self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4 + 8, convNum=4 + 3)
        self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.isTrain = train
        if warm_stage:
            for param in self.decoder.parameters():
                param.requires_grad = False

    def load_warm_checkpoint(self, checkpoint_path):
        _model = nn.parallel.DataParallel(ResHalf())
        pretrained_checkpoint = torch.load(checkpoint_path)
        _model.load_state_dict(pretrained_checkpoint['state_dict'], strict=False)
        _model = _model.module
        self.encoder = _model.encoder
        log(f'Loaded pretrained warm encoder from: {checkpoint_path}.')

    def forward(self, *x):
        # x[0]: color_image
        # x[1]: ref_halftone
        # noise = torch.randn_like(x[1]) * 0.3
        noise = torch.randn((x[0].size()[0], 1, x[0].size()[2], x[0].size()[3]),
                            dtype=x[0].dtype, device=x[0].device) * 0.3
        halfRes = self.encoder(torch.cat((x[0], noise), dim=1)) # type: ignore
        #halfRes = self.encoder(torch.cat((input_tensor+noise_map, input_tensor-noise_map), dim=1))
        halfResQ = self.quantizer(halfRes)
        #! for testing only
        #halfResQ = self.add_impluse_noise(halfResQ, p=0.20)
        restored = self.decoder(halfResQ)
        if self.isTrain:
            halfDCT = self.dcter(halfRes / 2. + 0.5)
            refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
            return halfRes, restored, halfDCT, refDCT, halfResQ
        else:
            return halfRes, restored, halfResQ

    def decode(self, x):
        # halfResQ = self.quantizer(x)
        restored = self.decoder(self.quantizer(x))
        # restored = self.decoder(x)
        return restored


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputX = ctx.saved_tensors
        return grad_output
