import torch
import numpy as np
from typing import Dict, Tuple, Union
from pytorch_msssim import ms_ssim


def spectral_angle_mapper(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-7,
    channels_first: bool = True,
    norm_threshold: float = 1e-10,
) -> Tuple[float, float]:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    if channels_first and a.ndim == 3:
        a = a.transpose(1, 2, 0)
        b = b.transpose(1, 2, 0)

    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])

    dot_product = np.sum(a_flat * b_flat, axis=1)
    norm_a = np.linalg.norm(a_flat, axis=1)
    norm_b = np.linalg.norm(b_flat, axis=1)

    valid_mask = (norm_a > norm_threshold) & (norm_b > norm_threshold)
    if not np.any(valid_mask):
        return float("nan"), float("nan")

    dot_product = dot_product[valid_mask]
    norm_a = norm_a[valid_mask]
    norm_b = norm_b[valid_mask]

    denom = norm_a * norm_b + eps
    cos_angle = np.clip(dot_product / denom, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)

    mean_rad = float(np.mean(angle_rad))
    mean_deg = mean_rad * 180.0 / np.pi
    return mean_rad, mean_deg


def _prepare_metric_inputs(
    a_np: np.ndarray,
    b_np: np.ndarray,
    a_torch: torch.Tensor,
    b_torch: torch.Tensor,
):
    """Select the metric scale from the channel configuration.

    The project uses 7 channels for Landsat-8 data normalized by 65535 and
    8 channels for Sentinel-2 data normalized by 10000. For 8-channel inputs,
    the historical paper evaluation rescales the normalized tensors by
    10000 / 65536 before PSNR and MS-SSIM calculation. This preserves the
    exact metric convention used for the reported Sentinel-2 results.
    """
    num_channels = int(a_np.shape[0])

    if num_channels == 8:
        scale_factor = 10000.0 / 65536.0
        return (
            a_np * scale_factor,
            b_np * scale_factor,
            a_torch * scale_factor,
            b_torch * scale_factor,
        )

    return a_np, b_np, a_torch, b_torch


def compute_multispectral_metrics(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    max_val: float = 1.0,
) -> Dict[str, float]:
    if isinstance(a, torch.Tensor):
        a_np = a.detach().cpu().numpy()
    else:
        a_np = np.asarray(a)

    if isinstance(b, torch.Tensor):
        b_np = b.detach().cpu().numpy()
    else:
        b_np = np.asarray(b)

    if a_np.ndim == 4:
        a_np = a_np[0]
        b_np = b_np[0]

    if a_np.ndim != 3 or b_np.ndim != 3:
        raise ValueError(
            f"Expected CHW or BCHW multispectral inputs, got {a_np.shape} and {b_np.shape}."
        )

    if a_np.shape != b_np.shape:
        raise ValueError(f"Input shapes must match, got {a_np.shape} and {b_np.shape}.")

    if isinstance(a, np.ndarray):
        a_torch = torch.from_numpy(a_np).float().unsqueeze(0)
        b_torch = torch.from_numpy(b_np).float().unsqueeze(0)
    else:
        a_torch = a[0:1] if a.ndim == 4 else a.unsqueeze(0)
        b_torch = b[0:1] if b.ndim == 4 else b.unsqueeze(0)

    metric_a_np, metric_b_np, metric_a_torch, metric_b_torch = _prepare_metric_inputs(
        a_np, b_np, a_torch, b_torch
    )

    mse = np.mean((metric_a_np - metric_b_np) ** 2)
    psnr = float("inf") if mse == 0 else (
        20 * np.log10(max_val) - 10 * np.log10(mse)
    )

    try:
        ms_ssim_val = ms_ssim(
            metric_a_torch,
            metric_b_torch,
            data_range=max_val,
        ).item()
    except Exception:
        ms_ssim_val = -1.0

    sam_rad, sam_deg = spectral_angle_mapper(
        a_np,
        b_np,
        channels_first=True,
    )

    return {
        "psnr": float(psnr),
        "ms_ssim": float(ms_ssim_val),
        "sam_rad": float(sam_rad),
        "sam_deg": float(sam_deg),
    }


def compute_sam_rad(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    channels_first: bool = True,
) -> float:
    sam_rad, _ = spectral_angle_mapper(a, b, channels_first=channels_first)
    return sam_rad
