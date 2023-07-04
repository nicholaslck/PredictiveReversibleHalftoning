"""
Utils for tensorboard
"""
from numpy import array
from logging import info as log
from torch.utils.tensorboard.writer import SummaryWriter

_writer: SummaryWriter = None  # type: ignore


def set_log_dir(log_dir=None):
    global _writer
    _writer = SummaryWriter(log_dir=log_dir)

def _get_metric_dict(metrics):
    return {
        'color_psnr': metrics[0],
        'color_psnr_stddev': metrics[1],
        'color_ssim': metrics[2],
        'color_ssim_stddev': metrics[3],
        'halftone_psnr': metrics[4],
        'halftone_psnr_stddev': metrics[5],
        'halftone_ssim': metrics[6],
        'halftone_ssim_stddev': metrics[7],
    }

def _get_hparams_dict(config):
    return {
        "quantize_loss_weight": config['loss_weights']['quantize_loss_weight'],
        "tone_loss_weight": config['loss_weights']['tone_loss_weight'],
        "structure_loss_weight": config['loss_weights']['structure_loss_weight'],
        "blueNoise_loss_weight": config['loss_weights']['blueNoise_loss_weight'],
        "vgg_loss_weight": config['loss_weights']['vgg_loss_weight'],
        "restore_loss_weight": config['loss_weights']['restore_loss_weight'],
        "feature_loss_weight": config['loss_weights']['feature_loss_weight']
    }

def log_hparams(config, metrics):
    # extract key-value pairs from config
    hparam_dict = _get_hparams_dict(config)

    # extract key-value pairs from metric
    metric_dict = _get_metric_dict(metrics)
    hparam_metric_dict = {}
    for key, value in metric_dict.items():
        hparam_metric_dict[f"hparam/{key}"] = value
    _writer.add_hparams(hparam_dict, hparam_metric_dict)

def log_epoch_metrics(epoch, metrics, to_tensorboard=True):

    log(f"================== Quantity results of epoch {epoch} =========================")
    log(f"PSNR color_pred <-> color_pred: {metrics[0]}")
    log(f"PSNR color_pred <-> color_pred stddev: {metrics[1]}")
    log(f"SSIM color_input <-> color_pred: {metrics[2]}")
    log(f"SSIM color_input <-> color_pred stddev: {metrics[3]}")
    log("--------------------------------------------------------------------------------")
    log(f"PSNR halftone <-> gray_input: {metrics[4]}")
    log(f"PSNR halftone <-> gray_input stddev: {metrics[5]}")
    log(f"SSIM halftone <-> gray_input: {metrics[6]}")
    log(f"SSIM halftone <-> gray_input stddev: {metrics[7]}")
    log("================================================================================")
    if to_tensorboard:
        metric_dict = _get_metric_dict(metrics)
        for key, value in metric_dict.items():
            _writer.add_scalar(f"validation/{key}", value, epoch, new_style=True)

def log_epoch_time(epoch, ts):
    _writer.add_scalar('epoch/time', ts, epoch)

def log_epoch_lr(epoch, lr):
    _writer.add_scalar('epoch/lr', lr, epoch)

def log_images(epoch, tag:str, img_t):
    _writer.add_images(tag, img_t, epoch, dataformats='NCHW')

def _get_metric_dict_PRLNet(metrics):
    return {
        'psnr': metrics[0],
        'psnr_stddev': metrics[1],
        'ssim': metrics[2],
        'ssim_stddev': metrics[3],
    }

def log_epoch_metrics_PRLNet(epoch, metrics, to_tensorboard=True):
    log(f"================== Quantity results of epoch {epoch} =========================")
    log(f"PSNR: {metrics[0]}")
    log(f"PSNR stddev: {metrics[1]}")
    log(f"SSIM: {metrics[2]}")
    log(f"SSIM stddev: {metrics[3]}")
    log("================================================================================")
    if to_tensorboard:
        metric_dict = _get_metric_dict_PRLNet(metrics)
        for key, value in metric_dict.items():
            _writer.add_scalar(f"validation/{key}", value, epoch, new_style=True)
