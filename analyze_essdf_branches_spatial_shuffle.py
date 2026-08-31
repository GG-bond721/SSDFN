#!/usr/bin/env python3
"""ESSDF branch correlation and perturbation-response analysis for SSDFN.

This script is designed to be copied into the SSDFN project root. It reuses the
same dataset class, model configuration, and checkpoint-loading convention as the
provided training/testing scripts. No model retraining is performed.

Analyses
--------
1. Pre-gating branch similarity at model.g_a.decouple:
   - spectral branch vs. spatial branch;
   - two fixed random channel halves within the spectral branch;
   - two fixed random channel halves within the spatial branch.
   Metrics: cosine similarity and Pearson correlation of channel-aggregated
   response maps, plus linear CKA on the full branch representations.

2. Controlled perturbation response:
   - channel-shared spatial shuffling applied to the common ESSDF input feature;
     the same spatial permutation is used for all channels, and the branch
     outputs are inverse-permuted before response measurement to remove trivial
     relocation effects;
   - mean-preserving band-wise scaling applied to the multispectral input to
     change relative spectral amplitudes.
   Response: ||F(x') - F(x)||_2 / (||F(x)||_2 + eps).

Two spatial probes are available:
   - within_block: shuffle spatial positions inside each non-overlapping block;
   - patch: shuffle whole non-overlapping blocks while preserving within-patch
     content. Use --spatial-shuffle-mode both to evaluate both probes.

The results support branch specialization only; they do not establish strict
statistical independence or perfectly pure spectral/spatial representations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Project imports. Run this file from the SSDFN project root.
from SSDFN import SSDFN
from utils.dataset_multispectral import MultispectralImageFolder


# -----------------------------------------------------------------------------
# Reproducibility, logging, and dataset helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("essdf_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def infer_image_names_from_dataloader(dataloader: DataLoader) -> Optional[List[str]]:
    """Infer deterministic image names from common dataset path attributes."""
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None

    candidate_attrs = (
        "samples",
        "imgs",
        "image_paths",
        "paths",
        "files",
        "filenames",
        "img_paths",
        "npy_files",
        "data_paths",
        "data",
    )

    for attr in candidate_attrs:
        if not hasattr(dataset, attr):
            continue
        value = getattr(dataset, attr)
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            continue

        names: List[str] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                item = item[0]
            if not isinstance(item, (str, os.PathLike)):
                names = []
                break
            names.append(Path(str(item)).stem)

        if len(names) == len(dataset):
            return names
    return None


def get_batch_tensor_and_names(
    batch: Any,
    batch_index: int,
    batch_size: int,
    inferred_names: Optional[Sequence[str]],
) -> Tuple[torch.Tensor, List[str]]:
    """Support tensor-only batches and (tensor, name/path, ...) batches."""
    if isinstance(batch, torch.Tensor):
        tensor = batch
        explicit_names = None
    elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
        tensor = batch[0]
        explicit_names = batch[1] if len(batch) >= 2 else None
    else:
        raise TypeError(f"Unsupported dataloader batch type: {type(batch)!r}")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected batch[0] to be a Tensor, got {type(tensor)!r}")

    names: List[str] = []
    if explicit_names is not None:
        if isinstance(explicit_names, str):
            explicit_names = [explicit_names]
        if isinstance(explicit_names, (list, tuple)) and len(explicit_names) == batch_size:
            for name in explicit_names:
                names.append(Path(str(name)).stem)

    if len(names) != batch_size:
        names = []
        start = batch_index * batch_size
        if inferred_names is not None and start + batch_size <= len(inferred_names):
            names = [str(inferred_names[start + j]) for j in range(batch_size)]
        else:
            names = [f"patch{start + j:06d}" for j in range(batch_size)]

    return tensor, names


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if multiple <= 1:
        return x, (0, 0)
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0), (pad_h, pad_w)


# -----------------------------------------------------------------------------
# Model loading and pre-gating feature extraction
# -----------------------------------------------------------------------------


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "network", "net"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                checkpoint = value
                break

    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping.")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "network.", "net."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        state_dict[new_key] = value

    if not state_dict:
        raise ValueError("No tensor parameters were found in the checkpoint.")
    return state_dict


def build_and_load_model(
    checkpoint_path: Path,
    num_channels: int,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[SSDFN, Dict[str, Any]]:
    # Exactly matches the model construction used in the provided test script.
    model = SSDFN(
        in_channels=num_channels,
        out_channels=num_channels,
        latent_channels=320,
        hyper_channels=640,
        slice_ch=[16, 16, 32, 64, 192],
        quant="ste",
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    required_keys = (
        "spectral_mix.conv3d_expand.weight",
        "g_a.initial_conv.weight",
        "g_a.down1.conv.weight",
        "g_a.multiscale1.preprocess.0.weight",
        "g_a.decouple.spectral_path.0.weight",
        "g_a.decouple.spectral_path.2.weight",
        "g_a.decouple.spatial_path.0.weight",
        "g_a.decouple.spatial_path.1.weight",
        "g_a.decouple.spatial_path.2.weight",
    )
    missing_required = [key for key in required_keys if key not in state_dict]
    if missing_required:
        raise RuntimeError(
            "Checkpoint is incompatible with the uploaded SSDFN encoder. "
            f"Missing required keys: {missing_required}"
        )

    # SSDFN.load_state_dict updates CompressAI buffers and then loads with
    # strict=False. Extra original-gating parameters are harmless because this
    # analysis uses only the pre-gating branch outputs.
    model.load_state_dict(state_dict)
    try:
        model.update(force=True)
    except Exception as exc:
        logger.warning(
            "model.update(force=True) failed, but entropy tables are not required "
            "for encoder feature extraction: %s",
            exc,
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "checkpoint_epoch": _metadata_to_python(
            checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None
        ),
        "checkpoint_loss": _metadata_to_python(
            checkpoint.get("loss") if isinstance(checkpoint, Mapping) else None
        ),
        "checkpoint_num_channels": _metadata_to_python(
            checkpoint.get("num_channels") if isinstance(checkpoint, Mapping) else None
        ),
    }

    logger.info("Loaded checkpoint: %s", checkpoint_path)
    logger.info("Checkpoint metadata: %s", metadata)
    return model, metadata


class ESSDFPreGateExtractor:
    """Extract the shared ESSDF input and pre-gating branch outputs."""

    def __init__(self, model: SSDFN):
        self.model = model

    @torch.inference_mode()
    def extract_essdf_input(self, x: torch.Tensor) -> torch.Tensor:
        """Follow the encoder exactly up to the input of g_a.decouple."""
        x = self.model.spectral_mix(x)
        h = self.model.g_a.initial_conv(x)
        h = self.model.g_a.down1(h)
        h = self.model.g_a.multiscale1(h)
        if not torch.isfinite(h).all():
            raise FloatingPointError("NaN or Inf detected in ESSDF input feature.")
        return h.float()

    @torch.inference_mode()
    def branches_from_essdf_input(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the two ESSDF branches on the same input feature."""
        essdf = self.model.g_a.decouple
        spectral = essdf.spectral_path(h)
        spatial = essdf.spatial_path(h)

        if spectral.shape != spatial.shape:
            raise RuntimeError(
                "ESSDF branch shape mismatch: "
                f"spectral={tuple(spectral.shape)}, spatial={tuple(spatial.shape)}"
            )
        if not torch.isfinite(spectral).all() or not torch.isfinite(spatial).all():
            raise FloatingPointError("NaN or Inf detected in ESSDF branch features.")
        return spectral.float(), spatial.float()

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.extract_essdf_input(x)
        return self.branches_from_essdf_input(h)


# -----------------------------------------------------------------------------
# Similarity metrics
# -----------------------------------------------------------------------------


def channel_aggregated_response(feature: torch.Tensor) -> torch.Tensor:
    """Return [B, H*W] mean-absolute response maps.

    Cosine/Pearson comparisons should not assume channel-to-channel alignment
    between independently learned branches. Channel aggregation avoids that
    unsupported assumption. Linear CKA below uses the full representations.
    """
    return feature.abs().mean(dim=1).flatten(start_dim=1)


def cosine_response_similarity(a: torch.Tensor, b: torch.Tensor, eps: float) -> np.ndarray:
    a_map = channel_aggregated_response(a)
    b_map = channel_aggregated_response(b)
    return F.cosine_similarity(a_map, b_map, dim=1, eps=eps).detach().cpu().numpy()


def pearson_response_correlation(a: torch.Tensor, b: torch.Tensor, eps: float) -> np.ndarray:
    a_map = channel_aggregated_response(a)
    b_map = channel_aggregated_response(b)
    a_centered = a_map - a_map.mean(dim=1, keepdim=True)
    b_centered = b_map - b_map.mean(dim=1, keepdim=True)
    numerator = (a_centered * b_centered).sum(dim=1)
    denominator = a_centered.norm(dim=1) * b_centered.norm(dim=1)
    value = numerator / denominator.clamp_min(eps)
    return value.detach().cpu().numpy()


def _feature_matrix_single(feature: torch.Tensor) -> torch.Tensor:
    """Convert one [C,H,W] feature tensor to [H*W,C]."""
    if feature.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(feature.shape)}")
    return feature.permute(1, 2, 0).reshape(-1, feature.shape[0]).float()


def linear_cka_single(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    max_samples: int,
    sample_seed: int,
) -> float:
    """Linear CKA with spatial positions as samples and channels as features."""
    x = _feature_matrix_single(a)
    y = _feature_matrix_single(b)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"CKA sample mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

    if max_samples > 0 and x.shape[0] > max_samples:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        index = torch.randperm(x.shape[0], generator=generator)[:max_samples].to(x.device)
        x = x.index_select(0, index)
        y = y.index_select(0, index)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    xy = x.transpose(0, 1) @ y
    xx = x.transpose(0, 1) @ x
    yy = y.transpose(0, 1) @ y

    numerator = torch.sum(xy.square())
    denominator = torch.sqrt(torch.sum(xx.square()) * torch.sum(yy.square())).clamp_min(eps)
    value = numerator / denominator
    return float(value.clamp(0.0, 1.0).item())


def compute_pair_metrics_batch(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
    cka_max_samples: int,
    seed_base: int,
) -> List[Dict[str, float]]:
    if a.shape[0] != b.shape[0]:
        raise ValueError("Batch-size mismatch in feature-pair metrics.")

    cosine_values = cosine_response_similarity(a, b, eps)
    pearson_values = pearson_response_correlation(a, b, eps)
    rows: List[Dict[str, float]] = []
    for index in range(a.shape[0]):
        cka = linear_cka_single(
            a[index],
            b[index],
            eps=eps,
            max_samples=cka_max_samples,
            sample_seed=seed_base + index,
        )
        rows.append(
            {
                "cosine_similarity": float(cosine_values[index]),
                "pearson_correlation": float(pearson_values[index]),
                "linear_cka": cka,
            }
        )
    return rows


def make_fixed_channel_splits(
    channels: int,
    repeats: int,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    if channels < 2:
        raise ValueError("At least two channels are required for internal analysis.")
    if repeats < 1:
        raise ValueError("split_repeats must be at least 1.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    half = channels // 2
    splits: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(repeats):
        permutation = torch.randperm(channels, generator=generator)
        left = permutation[:half]
        right = permutation[half:]
        splits.append((left, right))
    return splits


def compute_internal_metrics_batch(
    feature: torch.Tensor,
    splits: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    eps: float,
    cka_max_samples: int,
    seed_base: int,
) -> List[Dict[str, float]]:
    per_repeat: List[List[Dict[str, float]]] = []
    for repeat_index, (left_cpu, right_cpu) in enumerate(splits):
        left = left_cpu.to(feature.device)
        right = right_cpu.to(feature.device)
        metrics = compute_pair_metrics_batch(
            feature.index_select(1, left),
            feature.index_select(1, right),
            eps=eps,
            cka_max_samples=cka_max_samples,
            seed_base=seed_base + repeat_index * 100_000,
        )
        per_repeat.append(metrics)

    output: List[Dict[str, float]] = []
    for sample_index in range(feature.shape[0]):
        output.append(
            {
                key: float(np.mean([repeat[sample_index][key] for repeat in per_repeat]))
                for key in ("cosine_similarity", "pearson_correlation", "linear_cka")
            }
        )
    return output


# -----------------------------------------------------------------------------
# Perturbations and feature response
# -----------------------------------------------------------------------------


def gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Gaussian kernel size must be an odd integer >= 3.")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d


def apply_gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel2d(kernel_size, sigma, x.device)
    channels = x.shape[1]
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    pad = kernel_size // 2
    pad_mode = "reflect" if x.shape[-2] > pad and x.shape[-1] > pad else "replicate"
    padded = F.pad(x, (pad, pad, pad, pad), mode=pad_mode)
    return F.conv2d(padded, weight, groups=channels)



def _blockify_spatial(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int]:
    """Convert BCHW to [B,C,nH,nW,P], where P=block_size**2."""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    b, c, h, w = x.shape
    if block_size < 2:
        raise ValueError("spatial_block_size must be >= 2.")
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"ESSDF input spatial size {(h, w)} is not divisible by "
            f"block_size={block_size}. Choose a divisor of both dimensions."
        )
    nh = h // block_size
    nw = w // block_size
    blocks = (
        x.reshape(b, c, nh, block_size, nw, block_size)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(b, c, nh, nw, block_size * block_size)
    )
    return blocks, nh, nw


def _unblockify_spatial(
    blocks: torch.Tensor,
    block_size: int,
    nh: int,
    nw: int,
) -> torch.Tensor:
    """Inverse of _blockify_spatial."""
    b, c = blocks.shape[:2]
    return (
        blocks.reshape(b, c, nh, nw, block_size, block_size)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .reshape(b, c, nh * block_size, nw * block_size)
    )


def make_within_block_spatial_permutation(
    height: int,
    width: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create channel-shared permutations of positions inside each block.

    A different fixed permutation is generated for each non-overlapping block.
    The same permutation is applied to all feature channels. The returned
    inverse permutation can restore branch outputs to their original spatial
    coordinates before measuring response.
    """
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"Feature size {(height, width)} must be divisible by block_size={block_size}."
        )
    nh = height // block_size
    nw = width // block_size
    p = block_size * block_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # argsort of iid random keys gives one independent permutation per block.
    perm = torch.rand(nh, nw, p, generator=generator).argsort(dim=-1)
    inverse = perm.argsort(dim=-1)
    return perm, inverse


def apply_within_block_spatial_permutation(
    x: torch.Tensor,
    permutation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Shuffle positions inside each block identically for all channels."""
    blocks, nh, nw = _blockify_spatial(x, block_size)
    perm = permutation.to(x.device)
    if tuple(perm.shape) != (nh, nw, block_size * block_size):
        raise ValueError(
            f"Permutation shape {tuple(perm.shape)} is incompatible with "
            f"blocks {(nh, nw, block_size * block_size)}."
        )
    index = perm.view(1, 1, nh, nw, -1).expand(
        blocks.shape[0], blocks.shape[1], -1, -1, -1
    )
    shuffled = torch.gather(blocks, dim=-1, index=index)
    return _unblockify_spatial(shuffled, block_size, nh, nw)


def make_patch_permutation(
    height: int,
    width: int,
    block_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a fixed permutation of non-overlapping spatial blocks."""
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"Feature size {(height, width)} must be divisible by block_size={block_size}."
        )
    num_blocks = (height // block_size) * (width // block_size)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=generator)
    inverse = perm.argsort()
    return perm, inverse


def apply_patch_permutation(
    x: torch.Tensor,
    permutation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Shuffle whole blocks identically for all channels."""
    blocks, nh, nw = _blockify_spatial(x, block_size)
    b, c, _, _, p = blocks.shape
    flat = blocks.reshape(b, c, nh * nw, p)
    perm = permutation.to(x.device)
    if perm.numel() != nh * nw:
        raise ValueError(
            f"Patch permutation length {perm.numel()} does not match {nh*nw} blocks."
        )
    shuffled = flat.index_select(2, perm)
    shuffled = shuffled.reshape(b, c, nh, nw, p)
    return _unblockify_spatial(shuffled, block_size, nh, nw)


def apply_spatial_shuffle(
    x: torch.Tensor,
    mode: str,
    permutation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if mode == "within_block":
        return apply_within_block_spatial_permutation(x, permutation, block_size)
    if mode == "patch":
        return apply_patch_permutation(x, permutation, block_size)
    raise ValueError(f"Unsupported spatial shuffle mode: {mode}")


def make_spatial_shuffle_permutations(
    height: int,
    width: int,
    mode: str,
    block_size: int,
    repeats: int,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create fixed forward/inverse probes reused for all test images."""
    if repeats < 1:
        raise ValueError("spatial_shuffle_repeats must be >= 1.")
    probes: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for repeat_index in range(repeats):
        probe_seed = seed + repeat_index * 1009
        if mode == "within_block":
            probes.append(
                make_within_block_spatial_permutation(
                    height, width, block_size, probe_seed
                )
            )
        elif mode == "patch":
            probes.append(
                make_patch_permutation(height, width, block_size, probe_seed)
            )
        else:
            raise ValueError(f"Unsupported spatial shuffle mode: {mode}")
    return probes

def make_mean_preserving_scale_vectors(
    num_channels: int,
    delta: float,
    repeats: int,
    seed: int,
) -> List[torch.Tensor]:
    """Create fixed, mean-one band-scaling probes with factors in [1-d,1+d]."""
    if not (0.0 < delta < 1.0):
        raise ValueError("band_scale_delta must be in (0,1).")
    if repeats < 1:
        raise ValueError("perturb_repeats must be at least 1.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vectors: List[torch.Tensor] = []
    for _ in range(repeats):
        direction = torch.rand(num_channels, generator=generator) * 2.0 - 1.0
        direction = direction - direction.mean()
        max_abs = direction.abs().max().clamp_min(1e-8)
        direction = direction / max_abs
        factors = 1.0 + delta * direction
        # Numerical safeguard: preserve exact arithmetic mean 1.
        factors = factors / factors.mean()
        vectors.append(factors)
    return vectors


def apply_bandwise_scaling(x: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    if factors.numel() != x.shape[1]:
        raise ValueError(
            f"Scale-vector channels ({factors.numel()}) do not match input ({x.shape[1]})."
        )
    factors = factors.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * factors


def maybe_clamp(x: torch.Tensor, enabled: bool, low: float, high: float) -> torch.Tensor:
    if not enabled:
        return x
    return torch.clamp(x, min=low, max=high)


def relative_feature_response(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    eps: float,
) -> np.ndarray:
    if original.shape != perturbed.shape:
        raise ValueError(
            f"Feature-response shape mismatch: {tuple(original.shape)} vs {tuple(perturbed.shape)}"
        )
    numerator = (perturbed - original).flatten(start_dim=1).norm(dim=1)
    denominator = original.flatten(start_dim=1).norm(dim=1).clamp_min(eps)
    return (numerator / denominator).detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Statistics and outputs
# -----------------------------------------------------------------------------


@dataclass
class DescriptiveStats:
    n: int
    mean: float
    std: float
    sem: float
    ci95_low: float
    ci95_high: float


def descriptive_stats(values: Sequence[float]) -> DescriptiveStats:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return DescriptiveStats(0, math.nan, math.nan, math.nan, math.nan, math.nan)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    sem = std / math.sqrt(array.size) if array.size > 0 else math.nan
    half_width = 1.96 * sem
    return DescriptiveStats(
        n=int(array.size),
        mean=mean,
        std=std,
        sem=sem,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
    )


def one_sample_tests_greater(values: Sequence[float]) -> Dict[str, float]:
    """Test whether paired differences are greater than zero."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    result = {
        "paired_t_statistic": math.nan,
        "paired_t_p_one_sided": math.nan,
        "paired_t_p_two_sided": math.nan,
        "wilcoxon_statistic": math.nan,
        "wilcoxon_p_one_sided": math.nan,
        "cohen_dz": math.nan,
    }
    if array.size < 2:
        return result

    std = float(np.std(array, ddof=1))
    result["cohen_dz"] = float(np.mean(array) / std) if std > 0 else math.inf

    if scipy_stats is None:
        return result

    t_result = scipy_stats.ttest_1samp(array, popmean=0.0, nan_policy="omit")
    t_stat = float(t_result.statistic)
    p_two = float(t_result.pvalue)
    p_one = p_two / 2.0 if t_stat >= 0 else 1.0 - p_two / 2.0
    result.update(
        {
            "paired_t_statistic": t_stat,
            "paired_t_p_one_sided": p_one,
            "paired_t_p_two_sided": p_two,
        }
    )

    try:
        wilcoxon = scipy_stats.wilcoxon(array, alternative="greater", zero_method="wilcox")
        result["wilcoxon_statistic"] = float(wilcoxon.statistic)
        result["wilcoxon_p_one_sided"] = float(wilcoxon.pvalue)
    except Exception:
        pass
    return result


def summarize_correlation(
    per_image: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    metrics = ("cosine_similarity", "pearson_correlation", "linear_cka")
    for feature_pair, group in per_image.groupby("feature_pair", sort=False):
        for metric in metrics:
            stats = descriptive_stats(group[metric].to_numpy())
            summary_rows.append({"feature_pair": feature_pair, "metric": metric, **asdict(stats)})

    # Directly quantify the key claim: within-branch similarity > cross-branch similarity.
    pivot_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        pivot = per_image.pivot(index="image_index", columns="feature_pair", values=metric)
        required = {
            "spectral_vs_spatial",
            "spectral_internal_random_halves",
            "spatial_internal_random_halves",
        }
        if not required.issubset(pivot.columns):
            continue
        within_mean = 0.5 * (
            pivot["spectral_internal_random_halves"]
            + pivot["spatial_internal_random_halves"]
        )
        gap = within_mean - pivot["spectral_vs_spatial"]
        stats = descriptive_stats(gap.to_numpy())
        tests = one_sample_tests_greater(gap.to_numpy())
        pivot_rows.append(
            {
                "metric": metric,
                "contrast": "mean_within_branch_minus_cross_branch",
                **asdict(stats),
                **tests,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(pivot_rows)


def make_correlation_wide_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a paper-friendly table with one row per feature pair."""
    rows: List[Dict[str, Any]] = []
    metric_order = ("cosine_similarity", "pearson_correlation", "linear_cka")
    for feature_pair, group in summary.groupby("feature_pair", sort=False):
        row: Dict[str, Any] = {"feature_pair": feature_pair}
        for metric in metric_order:
            selected = group[group["metric"] == metric]
            if selected.empty:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_std"] = math.nan
            else:
                row[f"{metric}_mean"] = float(selected.iloc[0]["mean"])
                row[f"{metric}_std"] = float(selected.iloc[0]["std"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_perturbations(per_image: pd.DataFrame, eps: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for perturbation, group in per_image.groupby("perturbation", sort=False):
        spectral = group["spectral_response"].to_numpy(dtype=np.float64)
        spatial = group["spatial_response"].to_numpy(dtype=np.float64)

        if perturbation in ("within_block_spatial_shuffle", "patch_spatial_shuffle"):
            intended_name = "spatial_branch"
            intended = spatial
            other = spectral
        elif perturbation == "bandwise_scaling":
            intended_name = "spectral_branch"
            intended = spectral
            other = spatial
        else:
            intended_name = "unspecified"
            intended = spectral
            other = spatial

        gap = intended - other
        spectral_stats = descriptive_stats(spectral)
        spatial_stats = descriptive_stats(spatial)
        gap_stats = descriptive_stats(gap)
        tests = one_sample_tests_greater(gap)

        rows.append(
            {
                "perturbation": perturbation,
                "expected_more_sensitive_branch": intended_name,
                "spectral_mean": spectral_stats.mean,
                "spectral_std": spectral_stats.std,
                "spatial_mean": spatial_stats.mean,
                "spatial_std": spatial_stats.std,
                "expected_specialization_gap_mean": gap_stats.mean,
                "expected_specialization_gap_std": gap_stats.std,
                "expected_response_ratio": float(np.mean(intended) / (np.mean(other) + eps)),
                "n": gap_stats.n,
                "ci95_low": gap_stats.ci95_low,
                "ci95_high": gap_stats.ci95_high,
                **tests,
            }
        )
    return pd.DataFrame(rows)


def save_feature_map_figure(
    image_name: str,
    spectral: torch.Tensor,
    spatial: torch.Tensor,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_map = spectral[0].abs().mean(dim=0).detach().cpu().numpy()
    spatial_map = spatial[0].abs().mean(dim=0).detach().cpu().numpy()

    def normalize(array: np.ndarray) -> np.ndarray:
        low, high = np.percentile(array, [2, 98])
        return np.clip((array - low) / (high - low + 1e-8), 0.0, 1.0)

    spectral_map = normalize(spectral_map)
    spatial_map = normalize(spatial_map)
    difference = np.abs(spectral_map - spatial_map)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for axis, data, title in zip(
        axes,
        (spectral_map, spatial_map, difference),
        ("Spectral-oriented response", "Spatial-oriented response", "Absolute difference"),
    ):
        axis.imshow(data, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / f"{image_name}_branch_maps.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_summary(
    correlation_summary: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    if plt is None:
        return

    # Correlation summary: one panel per metric.
    metrics = ("cosine_similarity", "pearson_correlation", "linear_cka")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, metrics):
        subset = correlation_summary[correlation_summary["metric"] == metric]
        axis.bar(
            np.arange(len(subset)),
            subset["mean"].to_numpy(),
            yerr=subset["std"].to_numpy(),
            capsize=3,
        )
        axis.set_xticks(np.arange(len(subset)))
        axis.set_xticklabels(
            [
                "Cross" if name == "spectral_vs_spatial" else
                "Spectral\ninternal" if name.startswith("spectral_internal") else
                "Spatial\ninternal"
                for name in subset["feature_pair"]
            ]
        )
        axis.set_title(metric.replace("_", " "))
        axis.set_ylabel("Similarity")
    figure.tight_layout()
    figure.savefig(output_dir / "correlation_summary.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    # Perturbation responses.
    if not perturbation_summary.empty:
        labels = perturbation_summary["perturbation"].tolist()
        x = np.arange(len(labels))
        width = 0.36
        figure, axis = plt.subplots(figsize=(7, 4.2))
        axis.bar(
            x - width / 2,
            perturbation_summary["spectral_mean"],
            width,
            yerr=perturbation_summary["spectral_std"],
            label="Spectral-oriented branch",
            capsize=3,
        )
        axis.bar(
            x + width / 2,
            perturbation_summary["spatial_mean"],
            width,
            yerr=perturbation_summary["spatial_std"],
            label="Spatial-oriented branch",
            capsize=3,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_ylabel("Relative feature response")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "perturbation_summary.png", dpi=300, bbox_inches="tight")
        plt.close(figure)



def _metadata_to_python(value: Any) -> Any:
    """Convert common PyTorch/NumPy/path objects to plain Python values.

    This keeps metadata saving independent of JSON serialization and prevents
    scalar tensors such as checkpoint loss values from crashing the analysis at
    the very end of a long run.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _metadata_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_python(v) for v in value]
    return value


def save_metadata_txt(metadata: Mapping[str, Any], path: Path) -> None:
    """Save nested metadata as a readable UTF-8 TXT file.

    A plain-text format is intentionally used here because checkpoint metadata
    can contain tensors or NumPy scalar types that are not directly JSON
    serializable. Values are normalized to ordinary Python objects first.
    """
    normalized = _metadata_to_python(metadata)

    def write_mapping(file, mapping: Mapping[str, Any], indent: int = 0) -> None:
        prefix = "  " * indent
        for key, value in mapping.items():
            if isinstance(value, Mapping):
                file.write(f"{prefix}{key}:\n")
                write_mapping(file, value, indent + 1)
            elif isinstance(value, list):
                # Keep short scalar lists on one line; expand nested lists/dicts.
                if all(not isinstance(v, (list, dict)) for v in value):
                    file.write(f"{prefix}{key}: {value}\n")
                else:
                    file.write(f"{prefix}{key}:\n")
                    for index, item in enumerate(value):
                        if isinstance(item, Mapping):
                            file.write(f"{prefix}  [{index}]:\n")
                            write_mapping(file, item, indent + 2)
                        else:
                            file.write(f"{prefix}  [{index}]: {item}\n")
            else:
                file.write(f"{prefix}{key}: {value}\n")

    with path.open("w", encoding="utf-8") as file:
        write_mapping(file, normalized)


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def run_analysis(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    seed_everything(args.seed)

    if args.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Device: %s", device)
    logger.info("Dataset root: %s", args.dataset)
    logger.info("Checkpoint: %s", args.checkpoint)

    dataset = MultispectralImageFolder(
        args.dataset,
        split=args.split,
        transform=None,
        num_channels=args.num_channels,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    inferred_names = infer_image_names_from_dataloader(dataloader)

    model, checkpoint_metadata = build_and_load_model(
        Path(args.checkpoint), args.num_channels, device, logger
    )
    extractor = ESSDFPreGateExtractor(model)

    # Fixed probes are reused for all images, reducing probe-induced variance.
    scale_vectors = make_mean_preserving_scale_vectors(
        num_channels=args.num_channels,
        delta=args.band_scale_delta,
        repeats=args.perturb_repeats,
        seed=args.seed + 30_000,
    )

    correlation_records: List[Dict[str, Any]] = []
    perturbation_records: List[Dict[str, Any]] = []
    channel_splits: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    spatial_probe_cache: Dict[
        Tuple[int, int, str], List[Tuple[torch.Tensor, torch.Tensor]]
    ] = {}

    processed = 0
    feature_map_count = 0
    input_min = math.inf
    input_max = -math.inf

    for batch_index, batch in enumerate(dataloader):
        tensor, names = get_batch_tensor_and_names(
            batch=batch,
            batch_index=batch_index,
            batch_size=args.batch_size,
            inferred_names=inferred_names,
        )
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if tensor.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape={tuple(tensor.shape)}")
        if tensor.shape[1] != args.num_channels:
            raise ValueError(
                f"Dataset returned C={tensor.shape[1]}, expected {args.num_channels}."
            )
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Input batch contains NaN or Inf.")

        input_min = min(input_min, float(tensor.min().item()))
        input_max = max(input_max, float(tensor.max().item()))

        # Process samples one by one so each CSV row corresponds to one test image.
        for local_index in range(tensor.shape[0]):
            if args.max_images > 0 and processed >= args.max_images:
                break

            image_name = names[local_index]
            x_original = tensor[local_index : local_index + 1]
            original_hw = list(x_original.shape[-2:])
            x, padding = pad_to_multiple(x_original, args.pad_multiple)

            essdf_input = extractor.extract_essdf_input(x)
            spectral, spatial = extractor.branches_from_essdf_input(essdf_input)
            if channel_splits is None:
                if spectral.shape[1] != spatial.shape[1]:
                    raise RuntimeError("The two ESSDF branches must have equal channel counts.")
                channel_splits = make_fixed_channel_splits(
                    spectral.shape[1], args.split_repeats, args.seed + 20_000
                )

            cross = compute_pair_metrics_batch(
                spectral,
                spatial,
                eps=args.eps,
                cka_max_samples=args.cka_max_samples,
                seed_base=args.seed + processed * 1_000,
            )[0]
            spectral_internal = compute_internal_metrics_batch(
                spectral,
                channel_splits,
                eps=args.eps,
                cka_max_samples=args.cka_max_samples,
                seed_base=args.seed + 1_000_000 + processed * 10_000,
            )[0]
            spatial_internal = compute_internal_metrics_batch(
                spatial,
                channel_splits,
                eps=args.eps,
                cka_max_samples=args.cka_max_samples,
                seed_base=args.seed + 2_000_000 + processed * 10_000,
            )[0]

            for pair_name, metrics in (
                ("spectral_vs_spatial", cross),
                ("spectral_internal_random_halves", spectral_internal),
                ("spatial_internal_random_halves", spatial_internal),
            ):
                correlation_records.append(
                    {
                        "image": image_name,
                        "image_index": processed,
                        "feature_pair": pair_name,
                        "height": original_hw[0],
                        "width": original_hw[1],
                        "pad_h": padding[0],
                        "pad_w": padding[1],
                        **metrics,
                    }
                )

            # Controlled spatial probe at the common ESSDF input.
            #
            # Crucially, the SAME permutation is applied to every feature channel,
            # preserving the channel vector at each moved spatial position. After
            # each branch processes the shuffled feature, the inverse permutation
            # is applied to the branch output before measuring the response. This
            # removes the trivial effect that content has merely moved location.
            # A pointwise branch is spatial-permutation equivariant and should
            # therefore recover almost exactly after inverse alignment, whereas a
            # branch that uses 3x3 spatial neighborhoods can change because its
            # local neighborhoods were altered.
            spatial_modes = (
                ["within_block", "patch"]
                if args.spatial_shuffle_mode == "both"
                else [args.spatial_shuffle_mode]
            )
            h_feat, w_feat = essdf_input.shape[-2:]
            for spatial_mode in spatial_modes:
                cache_key = (h_feat, w_feat, spatial_mode)
                if cache_key not in spatial_probe_cache:
                    spatial_probe_cache[cache_key] = make_spatial_shuffle_permutations(
                        height=h_feat,
                        width=w_feat,
                        mode=spatial_mode,
                        block_size=args.spatial_block_size,
                        repeats=args.spatial_shuffle_repeats,
                        seed=args.seed + 40_000 + (0 if spatial_mode == "within_block" else 500_000),
                    )

                spectral_shuffle_responses: List[float] = []
                spatial_shuffle_responses: List[float] = []
                for forward_perm, inverse_perm in spatial_probe_cache[cache_key]:
                    shuffled_h = apply_spatial_shuffle(
                        essdf_input,
                        mode=spatial_mode,
                        permutation=forward_perm,
                        block_size=args.spatial_block_size,
                    )
                    spectral_shuffled, spatial_shuffled = extractor.branches_from_essdf_input(
                        shuffled_h
                    )

                    spectral_restored = apply_spatial_shuffle(
                        spectral_shuffled,
                        mode=spatial_mode,
                        permutation=inverse_perm,
                        block_size=args.spatial_block_size,
                    )
                    spatial_restored = apply_spatial_shuffle(
                        spatial_shuffled,
                        mode=spatial_mode,
                        permutation=inverse_perm,
                        block_size=args.spatial_block_size,
                    )

                    spectral_shuffle_responses.append(
                        float(
                            relative_feature_response(
                                spectral, spectral_restored, args.eps
                            )[0]
                        )
                    )
                    spatial_shuffle_responses.append(
                        float(
                            relative_feature_response(
                                spatial, spatial_restored, args.eps
                            )[0]
                        )
                    )

                mean_spectral_shuffle = float(np.mean(spectral_shuffle_responses))
                mean_spatial_shuffle = float(np.mean(spatial_shuffle_responses))
                perturbation_name = (
                    "within_block_spatial_shuffle"
                    if spatial_mode == "within_block"
                    else "patch_spatial_shuffle"
                )
                perturbation_records.append(
                    {
                        "image": image_name,
                        "image_index": processed,
                        "perturbation": perturbation_name,
                        "spectral_response": mean_spectral_shuffle,
                        "spatial_response": mean_spatial_shuffle,
                        "expected_specialization_gap": (
                            mean_spatial_shuffle - mean_spectral_shuffle
                        ),
                        "repeat_count": args.spatial_shuffle_repeats,
                        "parameters": json.dumps(
                            {
                                "level": "ESSDF_input_feature",
                                "mode": spatial_mode,
                                "block_size": args.spatial_block_size,
                                "inverse_aligned_before_response": True,
                                "channel_shared_permutation": True,
                                "repeat_count": args.spatial_shuffle_repeats,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )

            # Spectral probe: average multiple fixed mean-preserving scale vectors.
            spectral_scale_responses: List[float] = []
            spatial_scale_responses: List[float] = []
            for factors in scale_vectors:
                scaled = apply_bandwise_scaling(x_original, factors)
                scaled = maybe_clamp(
                    scaled,
                    args.clamp_perturbations,
                    args.clamp_min,
                    args.clamp_max,
                )
                scaled, _ = pad_to_multiple(scaled, args.pad_multiple)
                spectral_scaled, spatial_scaled = extractor(scaled)
                spectral_scale_responses.append(
                    float(relative_feature_response(spectral, spectral_scaled, args.eps)[0])
                )
                spatial_scale_responses.append(
                    float(relative_feature_response(spatial, spatial_scaled, args.eps)[0])
                )

            mean_spectral_scale = float(np.mean(spectral_scale_responses))
            mean_spatial_scale = float(np.mean(spatial_scale_responses))
            perturbation_records.append(
                {
                    "image": image_name,
                    "image_index": processed,
                    "perturbation": "bandwise_scaling",
                    "spectral_response": mean_spectral_scale,
                    "spatial_response": mean_spatial_scale,
                    "expected_specialization_gap": (
                        mean_spectral_scale - mean_spatial_scale
                    ),
                    "repeat_count": len(scale_vectors),
                    "parameters": json.dumps(
                        {
                            "delta": args.band_scale_delta,
                            "mean_preserving": True,
                            "repeat_count": len(scale_vectors),
                        },
                        ensure_ascii=False,
                    ),
                }
            )

            if feature_map_count < args.save_feature_maps:
                save_feature_map_figure(
                    image_name,
                    spectral,
                    spatial,
                    output_dir / "feature_maps",
                )
                feature_map_count += 1

            processed += 1
            if processed % args.progress_every == 0:
                logger.info("Processed %d images", processed)

        if args.max_images > 0 and processed >= args.max_images:
            break

    if processed == 0:
        raise RuntimeError("No test images were processed.")

    correlation_df = pd.DataFrame(correlation_records)
    perturbation_df = pd.DataFrame(perturbation_records)
    correlation_summary_df, correlation_contrast_df = summarize_correlation(correlation_df)
    correlation_wide_df = make_correlation_wide_table(correlation_summary_df)
    perturbation_summary_df = summarize_perturbations(perturbation_df, args.eps)

    correlation_df.to_csv(output_dir / "correlation_per_image.csv", index=False)
    correlation_summary_df.to_csv(output_dir / "correlation_summary.csv", index=False)
    correlation_wide_df.to_csv(output_dir / "correlation_summary_wide.csv", index=False)
    correlation_contrast_df.to_csv(output_dir / "correlation_contrast_summary.csv", index=False)
    perturbation_df.to_csv(output_dir / "perturbation_per_image.csv", index=False)
    perturbation_summary_df.to_csv(output_dir / "perturbation_summary.csv", index=False)

    with (output_dir / "paper_tables.txt").open("w", encoding="utf-8") as file:
        file.write("Correlation analysis (mean ± std)\n")
        for _, row in correlation_wide_df.iterrows():
            file.write(
                f"{row['feature_pair']}: "
                f"cosine={row['cosine_similarity_mean']:.6f} ± {row['cosine_similarity_std']:.6f}, "
                f"Pearson={row['pearson_correlation_mean']:.6f} ± {row['pearson_correlation_std']:.6f}, "
                f"CKA={row['linear_cka_mean']:.6f} ± {row['linear_cka_std']:.6f}\n"
            )
        file.write("\nPerturbation response (mean ± std)\n")
        for _, row in perturbation_summary_df.iterrows():
            file.write(
                f"{row['perturbation']}: "
                f"spectral={row['spectral_mean']:.6f} ± {row['spectral_std']:.6f}, "
                f"spatial={row['spatial_mean']:.6f} ± {row['spatial_std']:.6f}, "
                f"expected_gap={row['expected_specialization_gap_mean']:.6f} ± "
                f"{row['expected_specialization_gap_std']:.6f}, "
                f"p(one-sided)={row['paired_t_p_one_sided']:.6g}\n"
            )

    metadata = {
        "dataset": str(Path(args.dataset).resolve()),
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_images": processed,
        "num_channels": args.num_channels,
        "device": str(device),
        "input_range_observed": [input_min, input_max],
        "model_configuration": {
            "latent_channels": 320,
            "hyper_channels": 640,
            "slice_ch": [16, 16, 32, 64, 192],
            "quant": "ste",
        },
        "checkpoint_metadata": checkpoint_metadata,
        "feature_location": "g_a.decouple input and spectral_path/spatial_path outputs before gating/fusion",
        "cosine_pearson_representation": "mean absolute activation over channels; flattened spatial map",
        "cka_representation": "spatial positions as samples; channels as representation dimensions",
        "split_repeats": args.split_repeats,
        "cka_max_samples": args.cka_max_samples,
        "padding_multiple": args.pad_multiple,
        "spatial_shuffle": {
            "level": "common ESSDF input feature before branch processing",
            "mode": args.spatial_shuffle_mode,
            "block_size": args.spatial_block_size,
            "repeats": args.spatial_shuffle_repeats,
            "channel_shared_permutation": True,
            "inverse_aligned_before_response": True,
        },
        "bandwise_scaling": {
            "delta": args.band_scale_delta,
            "repeats": args.perturb_repeats,
            "mean_preserving": True,
            "scale_vectors": [vector.tolist() for vector in scale_vectors],
        },
        "clamp_perturbations": args.clamp_perturbations,
        "clamp_range": [args.clamp_min, args.clamp_max],
        "seed": args.seed,
        "scipy_available": scipy_stats is not None,
        "interpretation_warning": (
            "These measurements assess relative branch specialization and complementarity; "
            "they do not prove strict statistical independence or pure spectral/spatial separation."
        ),
    }
    # Save metadata as plain text instead of JSON. Checkpoint dictionaries may
    # contain scalar tensors (e.g., checkpoint_loss), which are not directly
    # JSON serializable and used to make an otherwise successful long run fail
    # after processing all images.
    metadata_path = output_dir / "analysis_metadata.txt"
    try:
        save_metadata_txt(metadata, metadata_path)
        logger.info("Metadata saved to: %s", metadata_path.resolve())
    except Exception as exc:
        # Metadata is auxiliary; never discard the already-computed CSV results
        # because of a metadata formatting problem.
        logger.warning("Failed to save metadata TXT: %s", exc)

    try:
        plot_summary(correlation_summary_df, perturbation_summary_df, output_dir)
    except Exception as exc:
        # Plotting is also auxiliary. The numerical CSV/TXT results remain the
        # authoritative outputs for the paper.
        logger.warning("Failed to generate summary plots: %s", exc)

    logger.info("Observed dataset tensor range: [%.6f, %.6f]", input_min, input_max)
    logger.info("Correlation summary:\n%s", correlation_summary_df.to_string(index=False))
    logger.info("Correlation paper table:\n%s", correlation_wide_df.to_string(index=False))
    logger.info("Correlation contrast summary:\n%s", correlation_contrast_df.to_string(index=False))
    logger.info("Perturbation summary:\n%s", perturbation_summary_df.to_string(index=False))
    logger.info("Results saved to: %s", output_dir.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pre-gating ESSDF branch similarity and responses to controlled "
            "spatial shuffling and mean-preserving band-wise scaling."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset root used by MultispectralImageFolder.")
    parser.add_argument("--checkpoint", required=True, help="Pretrained SSDFN checkpoint.")
    parser.add_argument("--output-dir", default="./essdf_branch_analysis")
    parser.add_argument("--split", default="test", help="Dataset split; normally test.")
    parser.add_argument("--num-channels", type=int, choices=(7, 8), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=200, help="0 means all test images.")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--pad-multiple", type=int, default=64)

    parser.add_argument(
        "--split-repeats",
        type=int,
        default=5,
        help="Number of fixed random channel-half partitions averaged per branch.",
    )
    parser.add_argument(
        "--cka-max-samples",
        type=int,
        default=2048,
        help="Maximum spatial positions per CKA computation; 0 uses all positions.",
    )

    parser.add_argument(
        "--spatial-shuffle-mode",
        choices=("within_block", "patch", "both"),
        default="within_block",
        help=(
            "Spatial perturbation at the common ESSDF input. 'within_block' "
            "shuffles positions inside each block; 'patch' shuffles whole blocks; "
            "'both' evaluates both."
        ),
    )
    parser.add_argument(
        "--spatial-block-size",
        type=int,
        default=8,
        help="Block size in ESSDF-input feature coordinates; must divide H and W.",
    )
    parser.add_argument(
        "--spatial-shuffle-repeats",
        type=int,
        default=5,
        help="Number of fixed spatial permutation probes averaged per image.",
    )
    parser.add_argument("--band-scale-delta", type=float, default=0.10)
    parser.add_argument(
        "--perturb-repeats",
        type=int,
        default=5,
        help="Number of fixed mean-preserving band-scaling vectors averaged per image.",
    )
    parser.add_argument(
        "--clamp-perturbations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp perturbed inputs to the model's normalized range.",
    )
    parser.add_argument("--clamp-min", type=float, default=0.0)
    parser.add_argument("--clamp-max", type=float, default=1.0)
    parser.add_argument("--save-feature-maps", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch_size != 1:
        parser.error("--batch-size must be 1 so every result row maps to one test image.")
    if args.max_images < 0:
        parser.error("--max-images must be >= 0.")
    if args.split_repeats < 1:
        parser.error("--split-repeats must be >= 1.")
    if args.spatial_block_size < 2:
        parser.error("--spatial-block-size must be >= 2.")
    if args.spatial_shuffle_repeats < 1:
        parser.error("--spatial-shuffle-repeats must be >= 1.")
    if args.perturb_repeats < 1:
        parser.error("--perturb-repeats must be >= 1.")
    if args.cka_max_samples < 0:
        parser.error("--cka-max-samples must be >= 0.")
    if args.save_feature_maps < 0:
        parser.error("--save-feature-maps must be >= 0.")

    run_analysis(args)


if __name__ == "__main__":
    main()
