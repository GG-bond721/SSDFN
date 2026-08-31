import os
import torch
import torch.nn.functional as F
import time
import re
import pandas as pd
import numpy as np
from PIL import Image
from utils.utils_multispectral import AverageMeter
from utils.metrics_multispectral import compute_multispectral_metrics

def get_stretch_params(band):
    return np.percentile(band, (2, 98))


def apply_stretch(band, low, high):
    band = np.clip(band, low, high)
    return ((band - low) / (high - low + 1e-8) * 255).astype(np.uint8)


def get_nir_index(num_channels):

    if num_channels == 7:
        return 4, "Landsat-8 NIR, index 4"
    if num_channels == 8:
        return 3, "Sentinel-2 NIR, index 3"
    return 0, "Fallback NIR index 0"


def to_float01_for_save(arr, name="array", logger=None):

    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"{name} should be a 2D NIR band, but got shape={arr.shape}.")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf before normalization.")

    raw_dtype = arr.dtype
    raw_min = float(np.nanmin(arr))
    raw_max = float(np.nanmax(arr))

    if np.issubdtype(arr.dtype, np.integer):
        arr_f = arr.astype(np.float32)
        if arr.dtype == np.uint8:
            arr_f = arr_f / 255.0
            scale_info = "uint8 -> divided by 255"
        elif arr.dtype == np.uint16:
            arr_f = arr_f / 65535.0
            scale_info = "uint16 -> divided by 65535"
        else:
            if raw_max > 1.5:
                arr_f = arr_f / 65535.0
                scale_info = f"{raw_dtype} DN-like -> divided by 65535"
            else:
                scale_info = f"{raw_dtype} integer [0,1] -> unchanged"
    else:
        arr_f = arr.astype(np.float32)
        if raw_max > 1.5:
            arr_f = arr_f / 65535.0
            scale_info = "float DN-like -> divided by 65535"
        else:
            scale_info = "float [0,1] -> unchanged"

    arr_f = np.clip(arr_f, 0.0, 1.0).astype(np.float32)

    if not np.isfinite(arr_f).all():
        raise ValueError(f"{name} contains NaN or Inf after normalization.")

    if logger is not None:
        logger.info(
            f"[NIR_SAVE] {name}: dtype={raw_dtype}, "
            f"raw_min={raw_min:.6f}, raw_max={raw_max:.6f}, "
            f"save_min={arr_f.min():.6f}, save_max={arr_f.max():.6f}, "
            f"{scale_info}"
        )

    return arr_f


def cuda_synchronize_if_needed(x=None):
    if torch.cuda.is_available():
        if x is None:
            torch.cuda.synchronize()
        elif isinstance(x, torch.Tensor) and x.is_cuda:
            torch.cuda.synchronize()

def infer_image_names_from_dataloader(test_dataloader):

    dataset = getattr(test_dataloader, "dataset", None)
    if dataset is None:
        return None

    candidate_attrs = [
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
    ]

    for attr in candidate_attrs:
        if not hasattr(dataset, attr):
            continue

        value = getattr(dataset, attr)

        if not isinstance(value, (list, tuple)):
            continue

        if len(value) == 0:
            continue

        names = []

        for item in value:
            if isinstance(item, (list, tuple)):
                item = item[0]

            if isinstance(item, (str, os.PathLike)):
                names.append(os.path.splitext(os.path.basename(str(item)))[0])
            else:
                names = []
                break

        if len(names) == len(dataset):
            return names

    return None


def get_batch_tensor_and_name(batch):

    if isinstance(batch, torch.Tensor):
        return batch, None

    if isinstance(batch, (list, tuple)):
        img = batch[0]

        name = None
        if len(batch) >= 2:
            maybe_name = batch[1]

            if isinstance(maybe_name, (list, tuple)):
                maybe_name = maybe_name[0]

            if isinstance(maybe_name, str):
                name = os.path.splitext(os.path.basename(maybe_name))[0]

        return img, name

    raise TypeError(f"Unsupported dataloader batch type: {type(batch)}")


def infer_target_bpp_from_string(text):

    if text is None:
        return np.nan

    text = str(text)

    m = re.search(r"bpp([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        value = m.group(1)

        # bpp08 / bpp09 这种写法
        if "." not in value and len(value) == 2 and value.startswith("0"):
            return float("0." + value[1])

        return float(value)

    return np.nan


def msssim_to_db(msssim_val):
    if msssim_val >= 1.0:
        return float("inf")
    return float(-10.0 * np.log10(1.0 - float(msssim_val)))


def save_full_multispectral_npy(arr_hwc, out_path):

    arr_hwc = np.asarray(arr_hwc).astype(np.float32)

    if arr_hwc.ndim != 3:
        raise ValueError(f"Full multispectral npy should be HWC, got shape={arr_hwc.shape}")

    if not np.isfinite(arr_hwc).all():
        raise ValueError(f"Full multispectral npy contains NaN or Inf: {out_path}")

    np.save(out_path, arr_hwc)


def compress_one_image_real_entropy(model, x, H, W):
    cuda_synchronize_if_needed(x)
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    cuda_synchronize_if_needed(x)
    enc_time = time.time() - start_time

    num_bytes = sum(len(s[0]) for s in out["strings"])
    num_bits = num_bytes * 8.0

    bpp = num_bits / (H * W)

    return out, bpp, enc_time


def decompress_one_image_real_entropy(model, out_comp, H, W):

    strings = out_comp["strings"]
    shape = out_comp["shape"]

    cuda_synchronize_if_needed()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape)
    cuda_synchronize_if_needed()
    dec_time = time.time() - start_time

    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, :H, :W]
    return x_hat, dec_time


def test_model_multispectral(
    test_dataloader,
    net,
    logger_test,
    save_dir,
    epoch,
    num_channels=7,
    method_name="ours",
    target_bpp=None,
    save_full_recon_npy=True,
    save_full_original_npy=False
):

    net.eval()
    device = next(net.parameters()).device

    if target_bpp is None:
        target_bpp = infer_target_bpp_from_string(save_dir)

    avg_psnr = AverageMeter()
    avg_ms_ssim = AverageMeter()
    avg_sam_rad = AverageMeter()
    avg_sam_deg = AverageMeter()
    avg_bpp = AverageMeter()
    avg_deocde_time = AverageMeter()
    avg_encode_time = AverageMeter()

    vis_dir = os.path.join(save_dir, "visualizations_gray")
    npy_dir = os.path.join(save_dir, "visualization_npy")

    full_recon_dir = os.path.join(save_dir, "full_reconstruction_npy")
    full_orig_dir = os.path.join(save_dir, "full_original_npy")

    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)

    if save_full_recon_npy:
        os.makedirs(full_recon_dir, exist_ok=True)

    if save_full_original_npy:
        os.makedirs(full_orig_dir, exist_ok=True)

    per_image_csv = os.path.join(save_dir, f"{method_name}_per_image_metrics.csv")
    per_image_txt = os.path.join(save_dir, f"{method_name}_per_image_metrics.txt")

    summary_csv = os.path.join(save_dir, f"{method_name}_experiment_summary.csv")
    summary_txt = os.path.join(save_dir, f"{method_name}_experiment_summary.txt")

    image_names = infer_image_names_from_dataloader(test_dataloader)

    if image_names is None:
        logger_test.info( "[WARNING] " )
    else:
        logger_test.info(f"Successfully inferred {len(image_names)} image names from dataset.")

    _, nir_name_default = get_nir_index(num_channels)
    logger_test.info(f"NIR setting by num_channels={num_channels}: {nir_name_default}")
    logger_test.info(f"method_name={method_name}, target_bpp={target_bpp}")
    logger_test.info(f"save_full_recon_npy={save_full_recon_npy}, save_full_original_npy={save_full_original_npy}")

    all_rows = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            img, batch_name = get_batch_tensor_and_name(batch)

            if batch_name is not None:
                image_name = batch_name
            elif image_names is not None and i < len(image_names):
                image_name = image_names[i]
            else:
                image_name = f"patch{i:06d}"

            img = img.to(device)
            B, C, H, W = img.shape

            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)

            out_comp, bpp, enc_time = compress_one_image_real_entropy(
                model=net,
                x=img_pad,
                H=H,
                W=W
            )

            x_hat, dec_time = decompress_one_image_real_entropy(
                model=net,
                out_comp=out_comp,
                H=H,
                W=W
            )

            x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
            x_hat = torch.clamp(x_hat, 0.0, 1.0)

            metrics = compute_multispectral_metrics(
                x_hat.squeeze(0),
                img.squeeze(0),
                max_val=1.0
            )

            psnr_value = float(metrics["psnr"])
            msssim_raw_value = float(metrics["ms_ssim"])
            msssim_db_value = msssim_to_db(msssim_raw_value)

            avg_psnr.update(psnr_value)
            avg_ms_ssim.update(msssim_raw_value)
            avg_sam_rad.update(metrics["sam_rad"])
            avg_sam_deg.update(metrics["sam_deg"])
            avg_bpp.update(bpp)
            avg_deocde_time.update(dec_time)
            avg_encode_time.update(enc_time)

            row = {
                "image": image_name,
                "method": method_name,
                "target_bpp": float(target_bpp) if np.isfinite(target_bpp) else np.nan,
                "qp": np.nan,
                "actual_bpp": float(bpp),
                "psnr": psnr_value,
                "msssim_raw": msssim_raw_value,
                "msssim_db": msssim_db_value,
                "msa": float(metrics["sam_deg"]),
                "msa_rad": float(metrics["sam_rad"])
            }

            all_rows.append(row)

            logger_test.info(
                f"Img[{i}] {image_name} | "
                f"Bpp:{bpp:.7f} | "
                f"PSNR:{psnr_value:.7f} | "
                f"MS-SSIM:{msssim_raw_value:.7f} | "
                f"SAM:{metrics['sam_deg']:.7f}° | "
                f"Enc:{enc_time:.6f}s Dec:{dec_time:.6f}s"
            )

            orig_np = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            recon_np = x_hat.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

            if save_full_recon_npy:
                save_full_multispectral_npy(
                    recon_np,
                    os.path.join(full_recon_dir, f"{image_name}_{method_name}_recon.npy")
                )

            if save_full_original_npy:
                save_full_multispectral_npy(
                    orig_np,
                    os.path.join(full_orig_dir, f"{image_name}_original.npy")
                )

            nir_idx, nir_name = get_nir_index(C)
            if nir_idx >= C:
                raise ValueError(f"NIR index {nir_idx} is out of range for C={C}.")

            nir_orig = orig_np[:, :, nir_idx]
            nir_recon = recon_np[:, :, nir_idx]

            nir_orig_f01 = to_float01_for_save(
                nir_orig,
                name=f"{image_name}_original_nir",
                logger=logger_test
            )

            nir_recon_f01 = to_float01_for_save(
                nir_recon,
                name=f"{image_name}_{method_name}_nir",
                logger=logger_test
            )

            orig_nir_path = os.path.join(npy_dir, f"{image_name}_original_nir.npy")
            if not os.path.exists(orig_nir_path):
                np.save(orig_nir_path, nir_orig_f01)

            np.save(
                os.path.join(npy_dir, f"{image_name}_{method_name}_nir.npy"),
                nir_recon_f01
            )

            np.save(
                os.path.join(npy_dir, f"{image_name}_{method_name}_nir_bpp{bpp:.4f}.npy"),
                nir_recon_f01
            )

            if method_name == "ours":
                np.save(
                    os.path.join(npy_dir, f"{image_name}_ssdfn_nir.npy"),
                    nir_recon_f01
                )

                np.save(
                    os.path.join(npy_dir, f"{image_name}_ssdfn_nir_bpp{bpp:.4f}.npy"),
                    nir_recon_f01
                )

            low, high = get_stretch_params(nir_orig_f01)
            stretched_orig = apply_stretch(nir_orig_f01, low, high)
            stretched_recon = apply_stretch(nir_recon_f01, low, high)

            orig_path = os.path.join(vis_dir, f"{image_name}_original.png")
            if not os.path.exists(orig_path):
                Image.fromarray(stretched_orig).convert("L").save(orig_path)

            Image.fromarray(stretched_recon).convert("L").save(
                os.path.join(vis_dir, f"{image_name}_{method_name}_bpp{bpp:.4f}.png")
            )

    if len(all_rows) > 0:
        df = pd.DataFrame(all_rows)

        columns = [
            "image",
            "method",
            "target_bpp",
            "qp",
            "actual_bpp",
            "psnr",
            "msssim_raw",
            "msssim_db",
            "msa",
            "msa_rad"
        ]

        df = df[columns]
        df = df.sort_values(by=["image", "actual_bpp"]).reset_index(drop=True)

        df.to_csv(per_image_csv, index=False)
        df.to_csv(per_image_txt, index=False, sep="\t")

        logger_test.info(f"Per-image metrics CSV saved to: {per_image_csv}")
        logger_test.info(f"Per-image metrics TXT saved to: {per_image_txt}")

        avg_msssim_raw_for_summary = float(df["msssim_raw"].mean())
        avg_msssim_db_for_summary = msssim_to_db(avg_msssim_raw_for_summary)

        summary = {
            "method": method_name,
            "target_bpp": float(target_bpp) if np.isfinite(target_bpp) else np.nan,
            "qp": np.nan,
            "actual_bpp": float(df["actual_bpp"].mean()),
            "psnr": float(df["psnr"].mean()),
            "msssim_raw": avg_msssim_raw_for_summary,
            "msssim_db": avg_msssim_db_for_summary,
            "msa": float(df["msa"].mean()),
            "msa_rad": float(df["msa_rad"].mean()),
            "image_count": int(len(df)),
            "enc_time": float(avg_encode_time.avg),
            "dec_time": float(avg_deocde_time.avg)
        }

        df_summary = pd.DataFrame([summary])

        summary_columns = [
            "method",
            "target_bpp",
            "qp",
            "actual_bpp",
            "psnr",
            "msssim_raw",
            "msssim_db",
            "msa",
            "msa_rad",
            "image_count",
            "enc_time",
            "dec_time"
        ]

        df_summary = df_summary[summary_columns]
        df_summary.to_csv(summary_csv, index=False)
        df_summary.to_csv(summary_txt, index=False, sep="\t")

        logger_test.info(f"Summary CSV saved to: {summary_csv}")
        logger_test.info(f"Summary TXT saved to: {summary_txt}")

    logger_test.info(
        f"Epoch:[{epoch}] | Avg Bpp:{avg_bpp.avg:.7f} | "
        f"Avg PSNR:{avg_psnr.avg:.7f} | "
        f"Avg MS-SSIM:{avg_ms_ssim.avg:.7f} | "
        f"Avg SAM:{avg_sam_deg.avg:.7f}° | "
        f"Enc:{avg_encode_time.avg:.7f}s | Dec:{avg_deocde_time.avg:.7f}s"
    )
