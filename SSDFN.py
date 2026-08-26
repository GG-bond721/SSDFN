import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste as ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder

from utils.func import update_registered_buffers, get_scale_table
from utils.ckbd import *
from layers import CheckboardMaskedConv2d
from entropyELIC import EntropyParametersEX
from contextELIC import ChannelContextEX
from CODEC import (AnalysisTransform,SynthesisTransform,HyperEncoder,HyperDecoder,SpectralMixModule)


class SSDFN(CompressionModel):

    def __init__(self,
                 in_channels=7,
                 out_channels=7,
                 latent_channels=320,
                 hyper_channels=640,
                 slice_ch=[16, 16, 32, 64, 192],
                 quant='ste',
                 **kwargs):
        super().__init__(entropy_bottleneck_channels=320, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.slice_ch = slice_ch
        self.slice_num = len(slice_ch)
        self.quant = quant

        assert sum(slice_ch) == latent_channels, \
            f"sum(slice_ch)={sum(slice_ch)} must equal latent_channels={latent_channels}"

        self.spectral_mix = SpectralMixModule(in_channels=in_channels)
        self.g_a = AnalysisTransform(
            in_channels=in_channels,out_channels=latent_channels

        )
        self.g_s = SynthesisTransform(
            in_channels=latent_channels,out_channels=out_channels
        )
        self.h_a = HyperEncoder(
            in_channels=latent_channels,out_channels=latent_channels
        )

        self.h_s = HyperDecoder()

        # (Checkerboard local context)
        self.local_context = nn.ModuleList([
            CheckboardMaskedConv2d(
                in_channels=c,
                out_channels=c * 2,
                kernel_size=5, stride=1, padding=2
            )
            for c in slice_ch
        ])

        # (Channel context)
        self.channel_context = nn.ModuleList([
            ChannelContextEX(
                in_dim=sum(slice_ch[:i]),
                out_dim=slice_ch[i] * 2,
                act=nn.ReLU
            ) if i > 0 else None
            for i in range(self.slice_num)
        ])

        self.entropy_params_anchor = nn.ModuleList([
            EntropyParametersEX(
                in_dim=hyper_channels,  # idx=0
                out_dim=slice_ch[i] * 2,
                act=nn.ReLU
            ) if i == 0 else EntropyParametersEX(
                in_dim=hyper_channels + slice_ch[i] * 2,  # idx>0: psi + channel_ctx
                out_dim=slice_ch[i] * 2,
                act=nn.ReLU
            )
            for i in range(self.slice_num)
        ])

        self.entropy_params_nonanchor = nn.ModuleList([
            EntropyParametersEX(
                in_dim=hyper_channels + slice_ch[i] * 2,  # idx=0: psi + local_ctx
                out_dim=slice_ch[i] * 2,
                act=nn.ReLU
            ) if i == 0 else EntropyParametersEX(
                in_dim=hyper_channels + slice_ch[i] * 4,  # idx>0: psi + local + channel
                out_dim=slice_ch[i] * 2,
                act=nn.ReLU
            )
            for i in range(self.slice_num)
        ])

        self.gaussian_conditional = GaussianConditional(None)


    def forward(self, x):

        x = self.spectral_mix(x)
        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if self.quant == 'ste':
            z_offset = self.entropy_bottleneck._get_medians()
            z_hat = ste_round(z - z_offset) + z_offset

        psi = self.h_s(z_hat)

        if psi.shape[2:] != y.shape[2:]:
            psi = F.interpolate(psi, size=y.shape[2:],
                                mode='bilinear', align_corners=False)

        y_slices = [
            y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:i + 1]), ...]
            for i in range(self.slice_num)
        ]

        y_hat_slices = []
        y_likelihood_slices = []

        for idx, y_slice in enumerate(y_slices):

            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:

                params_anchor = self.entropy_params_anchor[idx](psi)
            else:

                channel_ctx = self.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                params_anchor = self.entropy_params_anchor[idx](
                    torch.cat([channel_ctx, psi], dim=1)
                )

            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = F.softplus(scales_anchor) + 1e-4
            scales_anchor = ckbd_anchor(scales_anchor)
            means_anchor = ckbd_anchor(means_anchor)

            if self.quant == 'ste':
                slice_anchor = ste_round(slice_anchor - means_anchor) + means_anchor
            else:
                slice_anchor = self.gaussian_conditional.quantize(
                    slice_anchor, "noise" if self.training else "dequantize"
                )
            slice_anchor = ckbd_anchor(slice_anchor)

            local_ctx = self.local_context[idx](slice_anchor)

            if idx == 0:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, psi], dim=1)
                )
            else:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, psi], dim=1)
                )

            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = F.softplus(scales_nonanchor) + 1e-4
            scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
            means_nonanchor = ckbd_nonanchor(means_nonanchor)

            if self.quant == 'ste':
                slice_nonanchor = ste_round(slice_nonanchor - means_nonanchor) + means_nonanchor
            else:
                slice_nonanchor = self.gaussian_conditional.quantize(
                    slice_nonanchor, "noise" if self.training else "dequantize"
                )
            slice_nonanchor = ckbd_nonanchor(slice_nonanchor)

            scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
            means_slice = ckbd_merge(means_anchor, means_nonanchor)
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_slice, means_slice
            )

            y_hat_slice = slice_anchor + slice_nonanchor
            y_hat_slices.append(y_hat_slice)
            y_likelihood_slices.append(y_slice_likelihood)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood_slices, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_likelihoods": y_likelihoods,
                "z_likelihoods": z_likelihoods
            }
        }


    def compress(self, x):

        x = self.spectral_mix(x)
        y = self.g_a(x)
        z = self.h_a(y)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        psi = self.h_s(z_hat)
        if psi.shape[2:] != y.shape[2:]:
            psi = F.interpolate(psi, size=y.shape[2:],
                                mode='bilinear', align_corners=False)

        y_slices = [
            y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:i + 1]), ...]
            for i in range(self.slice_num)
        ]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_hat_slices = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)

            if idx == 0:
                params_anchor = self.entropy_params_anchor[idx](psi)
            else:
                channel_ctx = self.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                params_anchor = self.entropy_params_anchor[idx](
                    torch.cat([channel_ctx, psi], dim=1)
                )

            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = F.softplus(scales_anchor) + 1e-4
            slice_anchor = compress_anchor(
                self.gaussian_conditional, slice_anchor,
                scales_anchor, means_anchor,
                symbols_list, indexes_list
            )

            local_ctx = self.local_context[idx](slice_anchor)

            if idx == 0:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, psi], dim=1)
                )
            else:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, psi], dim=1)
                )

            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = F.softplus(scales_nonanchor) + 1e-4
            slice_nonanchor = compress_nonanchor(
                self.gaussian_conditional, slice_nonanchor,
                scales_nonanchor, means_nonanchor,
                symbols_list, indexes_list
            )

            y_hat_slices.append(slice_anchor + slice_nonanchor)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()

        torch.backends.cudnn.deterministic = False

        return {
            "strings": [[y_string], z_strings],
            "shape": z.size()[-2:]
        }

    def decompress(self, strings, shape):

        torch.backends.cudnn.deterministic = True
        torch.cuda.synchronize()
        start_time = time.process_time()

        y_strings = strings[0][0]
        z_strings = strings[1]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        psi = self.h_s(z_hat)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        y_hat_slices = []

        for idx in range(self.slice_num):
            # Anchor
            if idx == 0:
                params_anchor = self.entropy_params_anchor[idx](psi)
            else:
                channel_ctx = self.channel_context[idx](
                    torch.cat(y_hat_slices, dim=1)
                )
                params_anchor = self.entropy_params_anchor[idx](
                    torch.cat([channel_ctx, psi], dim=1)
                )

            scales_anchor, means_anchor = params_anchor.chunk(2, 1)
            scales_anchor = F.softplus(scales_anchor) + 1e-4
            slice_anchor = decompress_anchor(
                self.gaussian_conditional, scales_anchor, means_anchor,
                decoder, cdf, cdf_lengths, offsets
            )

            # Non-anchor
            local_ctx = self.local_context[idx](slice_anchor)

            if idx == 0:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, psi], dim=1)
                )
            else:
                params_nonanchor = self.entropy_params_nonanchor[idx](
                    torch.cat([local_ctx, channel_ctx, psi], dim=1)
                )

            scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
            scales_nonanchor = F.softplus(scales_nonanchor) + 1e-4
            slice_nonanchor = decompress_nonanchor(
                self.gaussian_conditional, scales_nonanchor, means_nonanchor,
                decoder, cdf, cdf_lengths, offsets
            )

            y_hat_slices.append(slice_anchor + slice_nonanchor)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)

        torch.backends.cudnn.deterministic = False
        torch.cuda.synchronize()
        end_time = time.process_time()

        return {
            "x_hat": x_hat,
            "cost_time": end_time - start_time
        }

    def update(self, scale_table=None, force=False):

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def load_state_dict(self, state_dict):

        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_offset", "_quantized_cdf", "_cdf_length"],
            state_dict,
        )

        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        nn.Module.load_state_dict(self, state_dict, strict=False)


def build_model(config_name='default', **kwargs):

    configs = {
        'default': {
            'in_channels': 7,
            'out_channels': 7,
            'latent_channels': 320,
            'hyper_channels': 640,
            'slice_ch': [16, 16, 32, 64, 192],
            'quant': 'ste'
        }
    }

    config = configs.get(config_name, configs['default'])
    config.update(kwargs)

    return SSDFN(**config)


