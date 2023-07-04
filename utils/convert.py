import torch
import numpy as np


def img2tensor(bgr_img: np.ndarray) -> torch.Tensor:
    if len(bgr_img.shape) == 2:
        bgr_img = bgr_img[..., np.newaxis]
    # from CHW to HWC
    img_t = bgr_img.transpose(2, 0, 1)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    return img_t


def tensor2img(bgr_img_t: torch.Tensor) -> np.ndarray:
    img = bgr_img_t.to('cpu').numpy()
    # from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = img[..., 0]
    return img


def denormalize_tensor(img_t: torch.Tensor) -> torch.Tensor:
    """ Denormalize tensor from range [-1, 1] to [0, 255] """
    return (img_t + 1.) * 127.5


def normalize_tensor(img_t: torch.Tensor) -> torch.Tensor:
    """ Normalize tensor from range [0, 255] to [-1, 1] """
    return img_t / 127.5 - 1.


def tensor_0_1(img_t: torch.Tensor) -> torch.Tensor:
    """ Normalize tensor from range [0, 255] to [0, 1]"""
    return img_t / 255.


def tensor_0_255(img_t: torch.Tensor) -> torch.Tensor:
    """ Normalize tensor from range [0, 1] to [0, 255]"""
    return img_t * 255.


RANGE_MODE_NEGATIVE_1_TO_1 = "RANGE_MODE_NEGATIVE_1_TO_1"
RANGE_MODE_0_TO_1 = "RANGE_MODE_0_TO_1"
RANGE_MODE_0_TO_255 = "RANGE_MODE_0_TO_255"

def range_cvt(t: torch.Tensor, from_mode: str, to_mode: str):
    if from_mode == RANGE_MODE_NEGATIVE_1_TO_1:
        __t_0_255 = denormalize_tensor(t)
        if to_mode == RANGE_MODE_0_TO_1:
            return tensor_0_1(__t_0_255)
        elif to_mode == RANGE_MODE_0_TO_255:
            return __t_0_255
        
    elif from_mode == RANGE_MODE_0_TO_1:
        __t_0_255 = tensor_0_255(t)
        if to_mode == RANGE_MODE_NEGATIVE_1_TO_1:
            return normalize_tensor(__t_0_255)
        elif to_mode == RANGE_MODE_0_TO_255:
            return __t_0_255

    elif from_mode == RANGE_MODE_0_TO_255:
        if to_mode == RANGE_MODE_NEGATIVE_1_TO_1:
            return normalize_tensor(t)
        elif to_mode == RANGE_MODE_0_TO_1:
            return tensor_0_1(t)
    else:
        raise ValueError(f"Invalid from_mode {from_mode}")
    raise ValueError(f"Invalid to_mode {to_mode}")

# Color space conversion:
# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
# https://www.easyrgb.com/en/math.php

def __split_dim(t: torch.Tensor):
    if len(t.shape) == 4:
        return 1
    else:
        return 0


def bgr2gray(bgr_t: torch.Tensor) -> torch.Tensor:
    """ 
    Input shape: (N,C,H,W)  
    Output shape: (N, C, H, W)
    Using CCIP 601:
    Y = 0.299 R + 0.587 G + 0.114 B
    https://en.wikipedia.org/wiki/Luma_%28video%29#Rec._601_luma_versus_Rec._709_luma_coefficients
     """
    b, g, r = torch.split(bgr_t.type(torch.float32), 1, dim=__split_dim(bgr_t))
    return b * 0.114 + g * 0.587 + r * 0.299

def gray2bgr(gray_t: torch.Tensor) -> torch.Tensor:
    y = torch.split(gray_t.type(torch.float32), 1, dim=__split_dim(gray_t))[0]
    bgr = y.expand(-1, 3, -1, -1)
    return bgr


def bgr2xyz(bgr_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    BGR range from 0 to 1
    dtype=torch.float32
    """
    b, g, r = torch.split(bgr_t.type(torch.float32), 1, dim=__split_dim(bgr_t))
    x = r * 0.412453 + g * 0.357580 + b * 0.180423
    y = r * 0.212671 + g * 0.715160 + b * 0.072169
    z = r * 0.019334 + g * 0.119193 + b * 0.950227
    return torch.cat([x, y, z], dim=__split_dim(bgr_t))


def xyz2lab(xyz_t: torch.Tensor) -> torch.Tensor:
    x, y, z = torch.split(xyz_t, 1, dim=__split_dim(xyz_t))
    # ref = [0.950456, 1., 1.088754]
    ref = [0.95047, 1., 1.08883]
    x = x / ref[0]
    y = y / ref[1]
    z = z / ref[2]

    # compute f(X), f(Y), f(Z)
    fx = torch.where(x > 0.008856, torch.pow(x, 1/3.), (7.787 * x) + (16 / 116.))
    fy = torch.where(y > 0.008856, torch.pow(y, 1/3.), (7.787 * y) + (16 / 116.))
    fz = torch.where(z > 0.008856, torch.pow(z, 1/3.), (7.787 * z) + (16 / 116.))

    l = (116. * fy) - 16.
    a = 500. * (fx - fy)
    b = 200. * (fy - fz)
    return torch.cat([l, a, b], dim=__split_dim(xyz_t))


def lab2xyz(lab_t: torch.Tensor) -> torch.Tensor:
    l, a, b = torch.split(lab_t, 1, dim=__split_dim(lab_t))
    # ref = [0.950456, 1., 1.088754]
    ref = [0.95047, 1., 1.08883]
    y = (l + 16) / 116.
    x = a / 500. + y
    z = y - b / 200.

    y = torch.where(torch.pow(y, 3) > 0.008856, torch.pow(y, 3), (y - 16 / 116.) / 7.787)
    x = torch.where(torch.pow(x, 3) > 0.008856, torch.pow(x, 3), (x - 16 / 116.) / 7.787)
    z = torch.where(torch.pow(z, 3) > 0.008856, torch.pow(z, 3), (z - 16 / 116.) / 7.787)

    x = x * ref[0]
    y = y * ref[1]
    z = z * ref[2]
    return torch.cat([x, y, z], dim=__split_dim(lab_t))


def xyz2bgr(xyz_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    Return: BGR range from 0 to 1
    dtype: torch.float32
    """
    x, y, z = torch.split(xyz_t, 1, dim=__split_dim(xyz_t))
    r = x * 3.240479 + y * -1.53715 + z * -0.498535
    g = x * -0.969256 + y * 1.875991 + z * 0.041556
    b = x * 0.055648 + y * -0.204043 + z * 1.057311
    return torch.cat([b, g, r], dim=__split_dim(xyz_t))


def bgr2lab(bgr_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    BGR range from 0 to 1
    dtype=torch.float32
    """
    return xyz2lab(bgr2xyz(bgr_t))


def lab2bgr(lab_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    Return: BGR range from 0 to 1
    dtype: torch.float32
    """
    return xyz2bgr(lab2xyz(lab_t))


def bgr2ycrcb(bgr_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    BGR range from 0 to 1
    dtype=torch.float32
    """
    b, g, r = torch.split(bgr_t, 1, dim=__split_dim(bgr_t))

    delta = 0.5  # because BGR range from 0 to 1
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + delta
    cb = (b - y) * 0.564 + delta
    return torch.cat([y, cr, cb], dim=__split_dim(bgr_t))


def ycrcb2bgr(ycrcb_t: torch.Tensor) -> torch.Tensor:
    """
    Input shape: (N,C,H,W)
    Return: BGR range from 0 to 1
    dtype: torch.float32
    """
    y, cr, cb = torch.split(ycrcb_t, 1, dim=__split_dim(ycrcb_t))

    delta = 0.5  # because BGR range from 0 to 1
    r = y + 1.403 * (cr - delta)
    g = y - 0.714 * (cr - delta) - 0.344 * (cb - delta)
    b = y + 1.773 * (cb - delta)
    return torch.cat([b, g, r], dim=__split_dim(ycrcb_t))

def bgr2rgb(bgr_t: torch.Tensor) -> torch.Tensor:
    b, g, r = torch.split(bgr_t.type(torch.float32), 1, dim=__split_dim(bgr_t))
    return torch.cat([r, g, b], dim=__split_dim(bgr_t))

def rgb2bgr(rgb_t: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.split(rgb_t.type(torch.float32), 1, dim=__split_dim(rgb_t))
    return torch.cat([b, g, r], dim=__split_dim(rgb_t))

# Todo test
if __name__ == '__main__':
    import cv2
    img_path = "assets/dad_bike.png"
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    bgr = img.transpose((2, 0, 1))
    bgr_t = torch.from_numpy(bgr.astype(np.float32))
    bgr_t = tensor_0_1(bgr_t)
    
    out_bgr = normalize_tensor(tensor_0_255(bgr_t)) # -1 to 1
    
    ycrcb_t = normalize_tensor(tensor_0_255(bgr2ycrcb(bgr_t))) # -1 to 1

    ycrcb_t = range_cvt(ycrcb_t, RANGE_MODE_NEGATIVE_1_TO_1, RANGE_MODE_0_TO_1)
    ycrcb_t = ycrcb2bgr(ycrcb_t)
    ycrcb_t = range_cvt(ycrcb_t, RANGE_MODE_0_TO_1, RANGE_MODE_NEGATIVE_1_TO_1)

    cv2.imwrite('bgr1.png', tensor2img(denormalize_tensor(out_bgr)))
    cv2.imwrite('bgr2.png', tensor2img(denormalize_tensor(ycrcb_t)))
