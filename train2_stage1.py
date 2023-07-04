import argparse
import logging
from logging import info as log
import os.path as path
import time
from datetime import datetime, timedelta
import yaml
import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from models.inverse_half import InverseHalf

import models.model as models
import utils.board as board
import utils.convert as convert
import utils.dataset as dataset
import utils.io as io
import utils.loss as loss
import utils.psnr as psnr
import utils.ssim as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
gaussian_blur = transforms.GaussianBlur(kernel_size=11, sigma=1.5)

# Directories initialization
checkpoint_dir = ""
debug = False

def init(config: dict):
    global checkpoint_dir
    global debug
    debug = config.get('debug') == True
    run_name = config['run_name']
    variation = config['variation']
    # Init directories
    if debug:
        output_dir = io.ensure_dir(path.join('output/train-debug', run_name, variation, current_time))
    else:
        output_dir = io.ensure_dir(path.join('output/train', run_name, variation, current_time))

    # Init tensorboard summary writer
    events_dir = io.ensure_dir(path.join(output_dir, 'events'))
    board.set_log_dir(events_dir)

    # Init stdout logging
    log_dir = io.ensure_dir(path.join(output_dir, 'logs'))
    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt='%y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(path.join(log_dir, f"{current_time}.txt")),
            logging.StreamHandler()])

    checkpoint_dir = io.ensure_dir(path.join(output_dir, 'checkpoints'))
    if debug:
        log("Debug mode activated!")
    log(f"output: {output_dir}")
    log(f"host: {socket.gethostname()}")
    log(f"config:\n\n{yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)}")


def train(config, resume_checkpoint=None):
    # Init training
    best_epoch = 0
    best_val_loss = torch.inf
    best_metrics = []
    epoch_start = 0
    epoch_end = config['training']['epochs']

    # Init model
    model = models.ResHalfPredictor(
        train=True,
        stage=1,
        encoder_pretrained=config['model']["reshalf_pretrained"],
        invhalf_pretrained=config['model']["invhalf_pretrained"],
        use_input_y=config['model']['use_input_y'],
        noise_weight=config['model']['noise_weight']
        )

    # GPU enable
    model = nn.parallel.DataParallel(model).to(device)

    # Init optimizer
    optimizer: optim.Optimizer = getattr(
        optim, config['optimizer']['type'])(
        model.parameters(),
        **config['optimizer']['option'])
    log(f"Optimizer {optimizer.__class__.__name__} loaded.")

    # Init LR scheduler
    lr_scheduler = getattr(
        optim.lr_scheduler, config['lr_scheduler']['type'])(
        optimizer, **config['lr_scheduler']['option'])
    log(f"LR Scheduler {lr_scheduler.__class__.__name__} loaded.")

    # Load init checkpoint state
    if config['model'].get('init_checkpoint') is not None:
        init_checkpoint_path = config['model']['init_checkpoint']
        init_checkpoint = torch.load(init_checkpoint_path)
        model.load_state_dict(init_checkpoint['model_state_dict'])
        log(f'Loaded init checkpoint {init_checkpoint_path}.')

    # Load checkpoint state
    if resume_checkpoint is not None and path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint)
        if checkpoint.get('epoch') is not None:
            epoch_start = checkpoint['epoch'] + 1
            log(f'Resume training from epoch {epoch_start}.')

        if checkpoint.get('model_state_dict') is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            log('Loaded model checkpoint.')

        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            log('Loaded optimizer checkpoint.')

        if checkpoint.get("lr_scheduler_state_dict") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            log('Loaded lr_sheduler checkpoint.')

        if checkpoint.get("best_val_loss") is not None:
            best_val_loss = checkpoint["best_val_loss"]
            log(f"Loaded best validation loss value {best_val_loss}")

        if checkpoint.get("best_metrics") is not None:
            best_metrics = checkpoint["best_metrics"]
            log('Loaded best validation metrics')

        if checkpoint.get("best_epoch") is not None:
            best_epoch = checkpoint["best_epoch"]
            log(f"Loaded best epoch {best_epoch}")

    # Init dataloaders
    dataset_root_dir = config['dataset']['root_dir']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    train_set = ConcatDataset([dataset.HalftoneVOC2012Training(root_dir=dataset_root_dir), dataset.ColorRampValidationDataset()])
    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log(f"Train DataLoader loaded with dataset: {train_set.__class__.__name__}")

    validate_set = dataset.HalftoneVOC2012Validation(root_dir=dataset_root_dir)
    validate_data_loader = DataLoader(validate_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log(f"Validate DataLoader loaded with dataset: {validate_set.__class__.__name__}")

    special_set = dataset.PlainColorTrainingDataset(root_dir=dataset_root_dir)
    special_data_loader = DataLoader(special_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    log(f"Special DataLoader loaded with dataset: {special_set.__class__.__name__}")

    # Init loss functions
    loss_func = {
        "quantize_loss": loss.bin_l1,
        "tone_loss": loss.gaussian_l2,
        "structure_loss": loss.ssim_loss,
        "vgg_loss": loss.Vgg19Loss(),
        "blue_noise_loss": loss.l1_loss,
        "restore_loss": loss.l2_loss,
        "feature_loss": loss.FeatureLoss(config['feature_loss_pretrained']),
    }

    # Run epochs
    epoch_count = 0
    metrics = []
    for epoch in range(epoch_start, epoch_end):
        if debug and epoch_count > 0:
            break
        log(f"Begin epoch {epoch}...")
        epoch_ts = time.time()
        epoch_lr = optimizer.state_dict()['param_groups'][0]['lr']

        train_loss = train_per_epoch(
            (epoch, epoch_end), config, model, optimizer=optimizer, train_loader=train_data_loader,
            special_loader=special_data_loader, loss_func=loss_func)
        val_loss, metrics = test_per_epoch(
            (epoch, epoch_end), config, model, test_data_loader=validate_data_loader, loss_func=loss_func)
        lr_scheduler.step(val_loss)

        # Save checkpoint state
        if bool(val_loss < best_val_loss):
            best_val_loss = val_loss
            best_metrics = metrics
            best_epoch = epoch
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_metrics': best_metrics,
            'best_epoch': best_epoch
        }
        torch.save(checkpoint_state, path.join(checkpoint_dir, "model_latest.pth.tar"))
        if epoch == best_epoch:
            torch.save(checkpoint_state, path.join(checkpoint_dir, "model_best.pth.tar"))
            torch.save(checkpoint_state, path.join(checkpoint_dir, f"model_epoch{format(epoch, '03')}.pth.tar"))

        # Std log
        epoch_time_delta = time.time() - epoch_ts
        estimated_remaining_time = epoch_time_delta * (epoch_end - epoch - 1)
        log(f"End epoch {epoch}.")
        log(f"--- epoch:{epoch}/{epoch_end - 1}")
        log(f"--- train_loss: {round(train_loss, 6)}")
        log(f"--- val_loss: {round(val_loss, 6)}")
        log(f"--- best_epoch: {best_epoch}")
        log(f"--- time consumed: {timedelta(seconds=epoch_time_delta)} | estimated remaining time: {timedelta(seconds=estimated_remaining_time)}")

        # Log to tensorboard
        board.log_epoch_time(epoch, epoch_time_delta)
        board.log_epoch_lr(epoch, epoch_lr)
        board.log_epoch_metrics(epoch, metrics)
        epoch_count += 1

    log(f"All training iterations are done!")
    log(f"+++ Best epoch {best_epoch} / {epoch_end - 1}.")
    log(f"+++ Best val_loss {best_val_loss}.")
    log(f"+++ Best metrics:")
    board.log_epoch_metrics(best_epoch, best_metrics, to_tensorboard=False)
    # board.log_hparams(config, best_metrics)


def train_per_epoch(epoch: tuple, config: dict, model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader,
                    special_loader: DataLoader, loss_func: dict):
    """Training iteration in a epoch

    Args:
        epoch (int, int): current epoch index and end epoch index
        config (dict): config dict
        model (nn.Module): model to be train
        optimizer (Optimizer): the training optimizer
        train_loader (DataLoader): dataloader for normal data
        special_loader (DataLoader): dataloader for special data
        loss_func (dict): dictionary contains loss functions
    """
    loss_weights = config['loss_weights']
    special_iter = iter(special_loader)
    total_batch = len(train_loader)
    total_loss = 0
    model.train()
    for batch_idx, (color_input, gray_input, cvt_input, halftone_input) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if debug and batch_idx > 0:
            break

        # Forward pass for normal data
        color_input = color_input.to(device)
        halftone_input = halftone_input.to(device)
        gray_input = gray_input.to(device)
        output = model(color_input, halftone_input)
        halftone = output[0]
        halftone_q = output[4]
        color_pred = output[1]
        y_pred = output[5]
        crcb_pred = output[6]

        # Loss for normal data
        quantize_loss = loss_func['quantize_loss'](halftone)
        tone_loss = loss_func['tone_loss'](halftone, color_input)
        structure_loss = loss_func['structure_loss'](halftone, color_input)
        vgg_loss = loss_func['vgg_loss'](color_pred / 2. + 0.5, color_input / 2. + 0.5)
        if config['training']['ycrcb_supervised'] == True:
            restore_loss = loss_func['restore_loss'](crcb_pred, cvt_input[0][:,1:,:,:].to(device))
        else:
            restore_loss = loss_func['restore_loss'](color_pred, color_input)
        feature_loss = loss_func['feature_loss'](halftone_q, gray_input)

        # Load special data
        try:
            sp_input = next(special_iter)
        except StopIteration:
            special_iter = iter(special_loader)
            sp_input = next(special_iter)

        # Forward pass for special color
        sp_color_input = sp_input[0].to(device)
        sp_halftone_input = sp_input[3].to(device)
        sp_output = model(sp_color_input, sp_halftone_input)
        """
        sp_output[0]: halftone
        sp_output[1]: restored color of special color
        sp_output[2]: DCT of halftone reference (ov)
        sp_output[3]: DCT of halftone
        sp_output[6]: crcb pred of halftone
        """

        # Loss for special color
        sp_quantize_loss = loss_func['quantize_loss'](sp_output[0])
        sp_tone_loss = loss_func['tone_loss'](sp_output[0], sp_color_input)
        blue_noise_loss = loss_func['blue_noise_loss'](sp_output[2], sp_output[3])
        _sp_restoreLoss = loss_func['restore_loss'](sp_output[1], sp_color_input)
        if config['training']['ycrcb_supervised'] == True:
            _sp_restoreLoss = loss_func['restore_loss'](sp_output[6], sp_input[2][0][:,1:,:,:].to(device))
        else:
            _sp_restoreLoss = loss_func['restore_loss'](sp_output[1], sp_color_input)

        loss = loss_weights["quantize_loss_weight"] * (0.5 * quantize_loss + 0.5 * sp_quantize_loss) \
            + loss_weights["tone_loss_weight"] * tone_loss \
            + loss_weights["structure_loss_weight"] * structure_loss \
            + loss_weights["blueNoise_loss_weight"] * (sp_tone_loss + blue_noise_loss) \
            + loss_weights["vgg_loss_weight"] * vgg_loss \
            + loss_weights["restore_loss_weight"] * (restore_loss + 0.2 * _sp_restoreLoss) \
            + loss_weights["feature_loss_weight"] * feature_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            log(f"[Train] >> epoch[{epoch[0]}/{epoch[1] - 1}] iter:{batch_idx + 1}/{total_batch} loss:{round(loss.item(), 6)}")

    return total_loss


def test_per_epoch(epoch: tuple, config: dict, model: nn.Module, test_data_loader: DataLoader, loss_func: dict):
    """Testing iteration in a epoch

    Args:
        epoch (int, int): current epoch index and end epoch index
        config (dict): config dict
        model (nn.Module): model to be inference
        test_data_loader (DataLoader): dataloader for normal data
        loss_func (dict): dictionary contains loss functions

    Returns:
        (float, [...]): the first value is the total loss, the second value is an array of metrics.
    """
    loss_weights = config['loss_weights']
    total_batch = len(test_data_loader)
    total_loss = 0
    all_metrics = [
        torch.zeros(total_batch, device=device, requires_grad=False),
        torch.zeros(total_batch, device=device, requires_grad=False),
        torch.zeros(total_batch, device=device, requires_grad=False),
        torch.zeros(total_batch, device=device, requires_grad=False)]
    model.eval()
    with torch.inference_mode():
        for batch_idx, (color_input, gray_input, cvt_input, halftone_input) in enumerate(test_data_loader):
            # We set this just for the example to run quickly.
            if debug and batch_idx > 0:
                break

            # Forward pass for normal data
            color_input = color_input.to(device)
            gray_input = gray_input.to(device)
            halftone_input = halftone_input.to(device)
            output = model(color_input, halftone_input)
            halftone = output[0]
            color_pred = output[1]
            halftone_q = output[4]

            # Loss calculation
            quantize_loss = loss_func['quantize_loss'](halftone)
            tone_loss = loss_func['tone_loss'](halftone, color_input)
            structure_loss = loss_func['structure_loss'](halftone, color_input)
            vgg_loss = loss_func['vgg_loss'](color_pred / 2. + 0.5, color_input / 2. + 0.5)
            restore_loss = loss_func['restore_loss'](color_pred, color_input)

            loss = loss_weights["quantize_loss_weight"] * quantize_loss \
                + loss_weights["tone_loss_weight"] * tone_loss \
                + loss_weights["structure_loss_weight"] * structure_loss \
                + loss_weights["vgg_loss_weight"] * vgg_loss \
                + loss_weights["restore_loss_weight"] * restore_loss
            total_loss += loss.item()

            # Metric calculation
            _color_input = convert.denormalize_tensor(color_input)
            _color_pred = convert.denormalize_tensor(color_pred)
            _halftone = convert.denormalize_tensor(halftone)
            _gray_input = convert.denormalize_tensor(gray_input)
            _halftone_q = convert.denormalize_tensor(halftone_q)
            _blur_halftone = gaussian_blur(_halftone)
            _blur_gray_input = gaussian_blur(_gray_input)

            psnr_restore = psnr.psnr(_color_pred, _color_input)
            ssim_restore = ssim.ssim(_color_pred, _color_input)
            psnr_halftone = psnr.psnr(_blur_halftone, _blur_gray_input)
            ssim_halftone = ssim.ssim(_halftone, _gray_input)
            all_metrics[0][batch_idx] = psnr_restore
            all_metrics[1][batch_idx] = ssim_restore
            all_metrics[2][batch_idx] = psnr_halftone
            all_metrics[3][batch_idx] = ssim_halftone

            if (batch_idx + 1) % 100 == 0:
                log(f"[Test] >> epoch[{epoch[0]}/{epoch[1] - 1}] iter:{batch_idx + 1}/{total_batch} loss:{round(loss.item(), 6)}")

            if batch_idx == 0:
                board.log_images(epoch[0], 'color_input', convert.tensor_0_1(convert.bgr2rgb(_color_input)))
                board.log_images(epoch[0], 'color_output', convert.tensor_0_1(convert.bgr2rgb(_color_pred)))
                board.log_images(epoch[0], 'halftone', convert.tensor_0_1(_halftone))
                board.log_images(epoch[0], 'halftone_q', convert.tensor_0_1(_halftone_q))

    metrics = [
        # psnr_restore
        all_metrics[0].mean(),
        all_metrics[0].std(),
        # ssim_restore
        all_metrics[1].mean(),
        all_metrics[1].std(),
        # psnr_halftone
        all_metrics[2].mean(),
        all_metrics[2].std(),
        # ssim_halftone
        all_metrics[3].mean(),
        all_metrics[3].std(),
    ]

    return total_loss, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-c', '--config', default=None, type=str, required=True)
    parser.add_argument('-r', '--resume', default=None, type=str, required=False)
    args = parser.parse_args()

    config: dict = {}
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        init(config)
        log("Program train2_stage1 start.")
        train(config, args.resume)
    except Exception:
        logging.error("Fatal Error in main loop", exc_info=True)
    finally:
        log("Program ended.")
