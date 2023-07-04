from datetime import datetime
import argparse
import os
import os.path as path
from logging import info as log
from torchvision.transforms import GaussianBlur

import cv2
import torch

import models.model as models
from utils.io import ensure_dir
import utils.convert as convert
import utils.psnr as psnr
import utils.ssim as ssim

cached_model = None

def get_model(method:str, device):
    if method == 'ours':
        model = models.ResHalfPredictor(train=False)
        model = torch.nn.parallel.DataParallel(model).to(device)
        checkpoint = 'checkpoints/ours_stage2.pth.tar'
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model_state_dict"], strict=True)
    elif method == 'reshalf':
        model = models.ResHalf(train=False)
        model = torch.nn.parallel.DataParallel(model).to(device)
        checkpoint = 'checkpoints/pretrained/reshalf_model_best.pth.tar'
        model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"], strict=True)
    else:
        raise ValueError(method)
    return model


def run(input_path: str, output_dir: str, method: str):
    output_dir = ensure_dir(output_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded image {input_path}.")
    print(f"Image shape: {img.shape}")

    model = get_model(method, device)

    gaussian_blur = GaussianBlur(kernel_size=11, sigma=1.5)

    with torch.inference_mode():
        img_t = convert.img2tensor(img).to(device)
        img_t = convert.normalize_tensor(img_t)

        img_t = torch.unsqueeze(img_t, 0)
        output = model(img_t, img_t)

        halftone_t = output[0]
        color_pred_t = output[1]
        halftone_q_t = output[2]

        halftone_t = convert.denormalize_tensor(halftone_t)
        halftone = convert.tensor2img(torch.squeeze(halftone_t, 0))
        cv2.imwrite(path.join(output_dir, 'output_halftone.png'), halftone)

        color_pred_t = convert.denormalize_tensor(color_pred_t)
        color_pred = convert.tensor2img(torch.squeeze(color_pred_t, 0))
        cv2.imwrite(path.join(output_dir, 'output_color.png'), color_pred)

        halftone_q_t = convert.denormalize_tensor(halftone_q_t)
        halftone_q = convert.tensor2img(torch.squeeze(halftone_q_t, 0))
        cv2.imwrite(path.join(output_dir, 'output_halftone_q.png'), halftone_q)

        color_input_t = convert.denormalize_tensor(img_t)
        gray_input_t = convert.img2tensor(gray_img).to(device).unsqueeze(0)

        _blur_halftone = gaussian_blur(halftone_t)
        _blur_gray_input = gaussian_blur(gray_input_t)
        psnr_restore = psnr.psnr(color_pred_t, color_input_t)
        ssim_restore = ssim.ssim(color_pred_t, color_input_t)
        psnr_halftone = psnr.psnr(_blur_halftone, _blur_gray_input)
        ssim_halftone = ssim.ssim(halftone_t, gray_input_t)

        print(f"================== Quantity results =========================")
        print(f"PSNR color_pred <-> color_pred: {psnr_restore.mean()}")
        # print(f"PSNR color_pred <-> color_pred stddev: {psnr_restore.std()}")
        print(f"SSIM color_input <-> color_pred: {ssim_restore.mean()}")
        # print(f"SSIM color_input <-> color_pred stddev: {ssim_restore.std()}")
        print("--------------------------------------------------------------------------------")
        print(f"PSNR halftone <-> gray_input: {psnr_halftone.mean()}")
        # print(f"PSNR halftone <-> gray_input stddev: {psnr_halftone.std()}")
        print(f"SSIM halftone <-> gray_input: {ssim_halftone.mean()}")
        # print(f"SSIM halftone <-> gray_input stddev: {ssim_halftone.std()}")
        print("================================================================================")

def decode(input_path: str, output_dir: str, method: str, gt: str):
    """ Given an halftone image, decode to a color image only """
    # Assume halftone image
    output_dir = ensure_dir(output_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    halftone = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print(f"Loaded image {input_path}.")
    print(f"Image shape: {halftone.shape}")

    model = get_model(method, device)


    with torch.inference_mode():
        halftone_t = convert.img2tensor(halftone).to(device)
        halftone_t = convert.normalize_tensor(halftone_t)
        halftone_t = torch.unsqueeze(halftone_t, 0)
        
        output = model.module.decode(halftone_t) # type: ignore
        color_pred_t = output[0] if type(output) is tuple else output

        color_pred_t = convert.denormalize_tensor(color_pred_t)
        color_pred = convert.tensor2img(torch.squeeze(color_pred_t, 0))
        cv2.imwrite(path.join(output_dir, 'output_color.png'), color_pred)


def generate_halftone(input_path: str, output_path:str, method: str, model=None, verbose=False):
    """ Given an color image, generate halftone image only """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Assume halftone image

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if verbose:
        print(f"Loaded image {input_path}")
        print(f"Image shape: {img.shape}")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if model is not None:
        _model = model
    else:
        _model = get_model(method, device)

    with torch.inference_mode():
        img_t = convert.img2tensor(img).to(device)
        img_t = convert.normalize_tensor(img_t)

        img_t = torch.unsqueeze(img_t, 0)
        output = _model(img_t, img_t)

        halftone_t = output[0]
        halftone_t = convert.denormalize_tensor(halftone_t)
        halftone = convert.tensor2img(torch.squeeze(halftone_t, 0))
        cv2.imwrite(output_path, halftone)
        

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser(description='Inference Fast')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--out_dir', default=f"inference-{current_time}", type=str, required=True, help='Output directory')
    # parser.add_argument('--checkpoint', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--method', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--decode_only', action='store_true', required=False)
    parser.add_argument('--gt', type=str, default="", required=False)
    args = parser.parse_args()

    if args.decode_only:
        decode(input_path=args.input, output_dir=args.out_dir, method=args.method, gt=args.gt)
    else:
        run(input_path=args.input, output_dir=args.out_dir, method=args.method)

    print("Inference done.")
