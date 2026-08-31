import torch

from utils.metrics_multispectral import compute_multispectral_metrics
from utils.utils_multispectral import AverageMeter


def test_one_epoch_multispectral(epoch, test_dataloader, model, criterion, save_dir,
                                 logger_val, tb_logger, num_channels=7):

    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    sam_rad = AverageMeter()
    sam_deg = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            if out_criterion["mse_loss"] is not None:
                mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])

            metrics = compute_multispectral_metrics(
                out_net['x_hat'].squeeze(0),
                d.squeeze(0),
                max_val=1.0
            )
            psnr.update(metrics['psnr'])
            ms_ssim.update(metrics['ms_ssim'])
            sam_rad.update(metrics['sam_rad'])
            sam_deg.update(metrics['sam_deg'])

    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)

    tb_logger.add_scalar('{}'.format('[val]: sam_rad'), sam_rad.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: sam_deg'), sam_deg.avg, epoch + 1)

    if out_criterion["mse_loss"] is not None:
        logger_val.info(
            f"Test ep {epoch}: "
            f"Loss: {loss.avg:.7f} | "
            f"Bpp: {bpp_loss.avg:.7f} | "
            f"PSNR: {psnr.avg:.7f} | "
            f"MS-SSIM: {ms_ssim.avg:.7f} | "
            f"SAM: {sam_deg.avg:.7f}°"
        )
        tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)

    if out_criterion["ms_ssim_loss"] is not None:
        logger_val.info(
            f"Test ep {epoch}: "
            f"Loss: {loss.avg:.7f} | "
            f"Bpp: {bpp_loss.avg:.7f} | "
            f"PSNR: {psnr.avg:.7f} | "
            f"MS-SSIM: {ms_ssim.avg:.7f} | "
            f"SAM: {sam_deg.avg:.7f}°"
        )
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)

    return loss.avg
