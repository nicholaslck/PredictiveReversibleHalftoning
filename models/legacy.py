# import logging
# from logging import info as log
# import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# from torch.autograd import Function
# # from .base import ResidualBlock, SkipConnection, ConvBlock

# from .inverse_half import InverseHalf

# from .submodel import Fusion, HourGlass
# from utils.dct import DCT_Lowfrequency
# from utils.convert import *

# class ResHalf_Double(ResHalf):
#     def __init__(self, train=True, warm_stage=False, encoder_pretrained=None):
#         super(ResHalf_Double, self).__init__()
#         self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
#         self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=8, convNum=8)
#         self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
#         # quantize [-1,1] data to be {-1,1}
#         self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
#         self.isTrain = train
#         if warm_stage:
#             for name, param in self.decoder.named_parameters():
#                 param.requires_grad = False
#         if encoder_pretrained:
#             __reshalf_model_pll = nn.parallel.DataParallel(ResHalf())
#             __dict = torch.load(encoder_pretrained)["state_dict"]
#             __reshalf_model_pll.load_state_dict(__dict, strict=False)
#             reshalf_model: ResHalf = __reshalf_model_pll.module
#             self.encoder = reshalf_model.encoder
#             logging.debug("ResHalf pretrained encoder loaded.")

#     def forward(self, *x):
#         # x[0]: color_image
#         # x[1]: ref_halftone
#         # noise = torch.randn_like(x[1]) * 0.3
#         noise = torch.randn((x[0].size()[0], 1, x[0].size()[2], x[0].size()[3]),
#                             dtype=x[0].dtype, device=x[0].device) * 0.3
#         halfRes = self.encoder(torch.cat((x[0], noise), dim=1))
#         halfResQ = self.quantizer(halfRes)
#         restored = self.decoder(halfResQ)
#         if self.isTrain:
#             halfDCT = self.dcter(halfRes / 2. + 0.5)
#             refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
#             return halfRes, halfDCT, refDCT, restored
#         else:
#             return halfRes, restored


# class ResHalfSuper(nn.Module):
#     def __init__(self, warm_stage=True, train=True):
#         super(ResHalfSuper, self).__init__()
#         self.encoder = HourGlass(
#             inChannel=4, outChannel=1, resNum=4, convNum=4)
#         self.inverse_half = InverseHalf()
#         self.decoder = HourGlass(
#             inChannel=2, outChannel=3, resNum=4, convNum=4)
#         self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
#         # quantize [-1,1] data to be {-1,1}
#         self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.

#         # if warm_stage:
#         #     for param in self.decoder.parameters():
#         #         param.requires_grad = False

#     def forward(self, *x):
#         # x[0]: color_image in shape (NCHW)
#         # x[1]: ref_halftone in shape (NCHW)
#         noise = torch.randn_like(x[1]) * 0.3
#         halftone = self.encoder(torch.cat((x[0], noise), dim=1))
#         halftone_q = self.quantizer(halftone)
#         gray_pred = self.inverse_half(halftone_q)
#         color_pred = self.decoder(torch.cat((halftone_q, gray_pred), dim=1))

#         halfDCT = self.dcter(halftone / 2. + 0.5)
#         refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))

#         return halftone, halfDCT, refDCT, color_pred, halftone_q, gray_pred


# class ResHalf_GrayToneAware(nn.Module):
#     def __init__(self) -> None:
#         super(ResHalf_GrayToneAware, self).__init__()

#         self.encoder = HourGlass(
#             inChannel=4, outChannel=1, resNum=4, convNum=4)

#         self.inverse_half = InverseHalf()

#         self.decoder = HourGlass(
#             inChannel=1,
#             outChannel=3,
#             resNum=4,
#             convNum=4)

#         self.refine_layer = HourGlass(
#             inChannel=4, outChannel=3, resNum=4, convNum=4)

#         self.dcter = DCT_Lowfrequency(size=256, fLimit=50)

#         # quantize [-1,1] data to be {-1,1}
#         self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.

#     def forward(self, *x):
#         # x[0]: color_image in shape (NCHW)
#         # x[1]: ref_halftone in shape (NCHW)
#         noise = torch.randn_like(x[1]) * 0.3
#         halftone = self.encoder(torch.cat((x[0], noise), dim=1))
#         halftone_q = self.quantizer(halftone)
#         gray_pred = self.inverse_half(halftone_q)
#         color_pred = self.decoder(halftone)
#         color_refine_pred = self.refine_layer(
#             torch.concat((gray_pred, color_pred), dim=1))

#         #  TODO: concat color_pred and gray_pred as final output

#         halfDCT = self.dcter(halftone / 2. + 0.5)
#         refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))

#         return halftone, halfDCT, refDCT, color_pred, halftone_q, gray_pred, color_refine_pred


# class ResHalf_GrayRes(nn.Module):
#     def __init__(self, warm_mode=False, train_mode=False) -> None:
#         super(ResHalf_GrayRes, self).__init__()

#         self.train_mode = train_mode
#         self.warm_mode = warm_mode

#         self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)

#         # self.fusion1 = ConvBlock(inChannels=2, outChannels=1, convNum=4)
#         self.fusion1 = Fusion(inChannel=2, outChannel=1)

#         self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)

#         # self.fusion2 = ConvBlock(inChannels=4, outChannels=3, convNum=4)
#         self.fusion2 = Fusion(inChannel=4, outChannel=3)

#         # quantize [-1,1] data to be {-1,1}
#         self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
#         self.dcter = DCT_Lowfrequency(size=256, fLimit=50)

#         if self.warm_mode:
#             for _, param in self.decoder.named_parameters():
#                 param.requires_grad = False
#             for _, param in self.fusion2.named_parameters():
#                 param.requires_grad = False

#     def forward(self, *x):
#         # x[0]: color_image in shape (NCHW)
#         # x[1]: gray_image in shape (NHW)
#         # x[2]: ref_halftone in shape (NCHW)
#         noise = torch.randn_like(x[1]) * 0.3

#         encoded = self.encoder(torch.cat((x[0], noise), dim=1))
#         halftone = self.fusion1(torch.cat((x[1], encoded), dim=1))
#         halftone_q = self.quantizer(halftone)
#         decoded = self.decoder(halftone_q)
#         color_pred = self.fusion2(torch.cat((x[1], decoded), dim=1))

#         if self.train_mode:
#             half_dct = self.dcter(halftone / 2. + 0.5)
#             ref_dct = self.dcter(bgr2gray(x[0] / 2. + 0.5))
#             return halftone, halftone_q, half_dct, ref_dct, color_pred
#         else:
#             return halftone, color_pred


# # class ResHalfSuper(nn.Module):
# #     """Some Information about ResHalfSuper"""
# #     def __init__(self, train=True, warm_stage=False):
# #         super(ResHalfSuper, self).__init__()
# #         self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
# #         self.inverse_half = InverseHalf()
# #         self.skip = SkipConnection(channels=1)
# #         self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)
# #         self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
# #         # quantize [-1,1] data to be {-1,1}
# #         self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
# #         self.isTrain = train
# #         if warm_stage:
# #             for _, param in self.decoder.named_parameters():
# #                 param.requires_grad = False

# #     # def add_impluse_noise(self, input_halfs, p=0.0):
# #     #     N,C,H,W = input_halfs.shape
# #     #     SNR = 1-p
# #     #     np_input_halfs = input_halfs.detach().to("cpu").numpy()
# #     #     np_input_halfs = np.transpose(np_input_halfs, (0, 2, 3, 1))
# #     #     for i in range(N):
# #     #         mask = np.random.choice((0, 1, 2), size=(H, W, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
# #     #         np_input_halfs[i, mask==1] = 1.0
# #     #         np_input_halfs[i, mask==2] = -1.0
# #     #     return torch.from_numpy(np_input_halfs.transpose((0, 3, 1, 2))).to(input_halfs.device)

# #     def forward(self, *x):
# #         """ x[0]: color_image, x[1]: ref_halftone  """

# #         noise = torch.randn_like(x[1]) * 0.3
# #         halftone = self.encoder(torch.cat((x[0], noise), dim=1))

# #         halftone_q = self.quantizer(halftone)

# #         gray_scale = self.inverse_half(halftone_q)

# #         combined = self.skip(halftone_q, gray_scale)

# #         restored_color = self.decoder(combined)

# #         if self.isTrain:
# #             halfDCT = self.dcter(halftone / 2. + 0.5)
# #             refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
# #             return halftone, halfDCT, refDCT, restored_color
# #         else:
# #             return halftone, restored_color
