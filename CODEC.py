import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from compressai.layers import GDN

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * torch.sigmoid(gate) * gate


class IGDN(GDN):
    def __init__(self, channels, **kwargs):
        super().__init__(channels, inverse=True, **kwargs)


class SpectralMixModule(nn.Module):
    def __init__(self, in_channels=7, expansion_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = in_channels * expansion_factor
        self.conv3d_expand = nn.Conv3d(
            1, self.mid_channels * 2,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            bias=True
        )
        self.swiglu = SwiGLU()
        self.conv3d_compress = nn.Conv3d(
            self.mid_channels, 1,
            kernel_size=(1, 1, 3),
            padding=(0, 0, 1),
            bias=True
        )
        self.use_ln = True
        if self.use_ln:
            pass

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_3d = x.unsqueeze(2).permute(0, 2, 3, 4, 1)
        x_3d = self.conv3d_expand(x_3d)
        if self.use_ln:
            x_3d = F.layer_norm(x_3d, x_3d.shape[1:])
        x_3d = self.swiglu(x_3d)
        x_3d = self.conv3d_compress(x_3d)
        if self.use_ln:
            x_3d = F.layer_norm(x_3d, x_3d.shape[1:])
        output = x_3d.permute(0, 4, 1, 2, 3).squeeze(2)

        return residual + output


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)


class DownGDN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 2 * channels, 3, padding=1)
        self.norm = nn.GroupNorm(1, 2 * channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(2 * channels, channels, 3, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = out + res
        return out


class MultiScaleFusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1)
        )
        branch_channels = out_channels // 3
        self.branch_a = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, groups=in_channels),
            nn.Conv2d(in_channels, branch_channels, 1)
        )
        self.branch_b = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels),
            nn.Conv2d(in_channels, branch_channels, 1)
        )
        self.branch_c = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4, groups=in_channels),
            nn.Conv2d(in_channels, out_channels - 2 * branch_channels, 1)
        )
        self.shuffle = ChannelShuffle(groups=4)
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.rb = ResidualBlock(out_channels)
        self.match_channels = None
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        assert not torch.isnan(x).any(), "NaN input to MSFB"
        x = self.preprocess(x)
        identity = x
        a = self.branch_a(x)
        b = self.branch_b(x)
        c = self.branch_c(x)
        out = torch.cat([a, b, c], dim=1)
        out = self.shuffle(out)
        out = self.fusion(out)
        out = self.rb(out)
        if self.match_channels is not None:
            identity = self.match_channels(identity)
        out = out + identity

        return out


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C)
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_norm = self.ln(x_perm)  # LN over last dim (C)
        x = x_norm.permute(0, 3, 1, 2).contiguous()  # back to (B, C, H, W)
        return x


class SpectralSpatialDecoupling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 2
        self.spectral_path = nn.Sequential(
            nn.Conv2d(channels, mid_channels * 2, 1),
            SwiGLU(),
            nn.Conv2d(mid_channels, mid_channels, 1),
            GDN(mid_channels)
        )
        self.spatial_path = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, mid_channels, 1),
            GDN(mid_channels)
        )
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.fusion = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        identity = x
        spectral = self.spectral_path(x)
        spatial = self.spatial_path(x)
        gate = self.gate(x)
        gate_spectral = gate[:, :gate.size(1) // 2, :, :]
        gate_spatial = gate[:, gate.size(1) // 2:, :, :]
        weighted_spectral = gate_spectral * spectral
        weighted_spatial = gate_spatial * spatial
        out = torch.cat([weighted_spectral, weighted_spatial], dim=1)
        #out = torch.cat([spectral, spatial], dim=1)
        out = self.fusion(out)
        out = out + identity
        return out


class PMESSDF(nn.Module):
    """
    Parameter-Matched ESSDF-style block.

    Purpose:
        Construct a homogeneous encoder-decoder baseline with the same
        topology and internal width on both sides, while approximately
        matching the total parameter budget of the proposed
        ESSDF + DSSDF configuration.

    Args:
        channels: input/output channels, default 320.
        branch_channels: width of each spectral/spatial branch.
                         For C=320, branch_channels=316 approximately
                         matches the parameter count of ESSDF + DSSDF.
        inverse:
            False -> GDN, for encoder / analysis transform.
            True  -> IGDN, for decoder / synthesis transform.
    """

    def __init__(
        self,
        channels=320,
        branch_channels=316,
        inverse=False,
    ):
        super().__init__()

        self.channels = channels
        self.branch_channels = branch_channels
        self.inverse = inverse
        Norm = IGDN if inverse else GDN
        self.spectral_path = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels * 2,
                kernel_size=1,
                bias=True
            ),
            SwiGLU(),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=1,
                bias=True
            ),
            Norm(branch_channels)
        )
        self.spatial_path = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True
            ),
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=1,
                bias=True
            ),
            Norm(branch_channels)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels * 2,
                kernel_size=1,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(
            branch_channels * 2,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        identity = x
        spectral = self.spectral_path(x)
        spatial = self.spatial_path(x)
        gate = self.gate(x)
        gate_spectral, gate_spatial = gate.chunk(2, dim=1)

        weighted_spectral = gate_spectral * spectral
        weighted_spatial = gate_spatial * spatial
        out = torch.cat(
            [weighted_spectral, weighted_spatial],
            dim=1
        )
        out = self.fusion(out)
        out = out + identity

        return out


class SpectralSpatialFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid_channels = channels // 2
        self.spectral_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, mid_channels, 1),
            IGDN(mid_channels)
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, mid_channels, 1),
            IGDN(mid_channels)
        )
        self.cross_attention = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            SwiGLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        identity = x
        spectral = self.spectral_branch(x)
        spatial = self.spatial_branch(x)
        concat = torch.cat([spectral, spatial], dim=1)
        attn = self.cross_attention(concat)
        enhanced = concat * attn
        #enhanced = concat
        out = self.fusion(enhanced)
        out = out + identity
        return out


class StandardResBlock(nn.Module):
    def __init__(self, channels, num_groups=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        return out


class AnalysisTransform(nn.Module):

    def __init__(self, in_channels=7, out_channels=320):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 320, 3, padding=1)
        self.down1 = DownGDN(320, 320)
        self.multiscale1 = MultiScaleFusionBlock(320, 320)
        #self.multiscale1 = StandardResBlock(320)
        self.decouple = SpectralSpatialDecoupling(320)
        #self.decouple = SwappedDSSDFEncoder( channels=320 )
        #self.decouple = PMESSDF(channels=320,branch_channels=316,inverse=False)
        #self.decouple = StandardResBlock(320)
        #self.fusion = SpectralSpatialFusion(320)
        self.down2 = DownGDN(320, 320)
        self.multiscale2 = MultiScaleFusionBlock(320, 320)
        #self.multiscale2 =StandardResBlock(320)
        self.down3 = DownGDN(320, 320)
        self.multiscale3 = MultiScaleFusionBlock(320, 320)
        #self.multiscale3 = StandardResBlock(320)
        self.down4 = DownGDN(320, 320)
        self.refine = nn.Conv2d(320, out_channels, 1)

    def forward(self, x):

        out = self.initial_conv(x)
        out = self.down1(out)
        out = self.multiscale1(out)
        out = self.decouple(out)
        #out = self.fusion(out)
        out = self.down2(out)
        out = self.multiscale2(out)
        out = self.down3(out)
        out = self.multiscale3(out)
        out = self.down4(out)
        y = self.refine(out)

        return y


class SynthesisTransform(nn.Module):
    def __init__(self, in_channels=320, out_channels=7):
        super().__init__()
        self.refine = nn.Conv2d(in_channels, 320, 1)
        self.up1 = nn.ConvTranspose2d(320, 320, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.multiscale1 = MultiScaleFusionBlock(320, 320)
        #self.multiscale1 = StandardResBlock(320)
        self.up2 = nn.ConvTranspose2d(320, 320, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.multiscale2 = MultiScaleFusionBlock(320, 320)
        #self.multiscale2 = StandardResBlock(320)
        self.fusion = SpectralSpatialFusion(320)
        #self.fusion = SwappedESSDFDecoder(channels=320)
        #self.decouple = SpectralSpatialDecoupling(320)
        #self.fusion = PMESSDF(channels=320,branch_channels=316,inverse=True)
        #self.fusion = StandardResBlock(320)
        self.up3 = nn.ConvTranspose2d(320, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.multiscale3 = MultiScaleFusionBlock(256, 256)
        #self.multiscale3 = StandardResBlock(256)
        self.up4 = nn.ConvTranspose2d(256, 192, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.output_conv = nn.Conv2d(192, out_channels, 3, padding=1)

    def forward(self, y_hat):
        out = self.refine(y_hat)
        out = self.up1(out)
        out = self.multiscale1(out)
        out = self.up2(out)
        out = self.multiscale2(out)
        out = self.fusion(out)
        #out = self.decouple(out)
        out = self.up3(out)
        out = self.multiscale3(out)
        out = self.up4(out)
        x_hat = self.output_conv(out)

        return x_hat


class HyperEncoder(nn.Module):
    def __init__(self, in_channels=320, out_channels=320):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 320, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(320, 192, 5, stride=2, padding=2)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(192, 192, 5, stride=2, padding=2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(192, 192, 5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(192, out_channels, 3, stride=1, padding=1)


    def forward(self, y):
        x = self.conv1(y)
        x = self.lrelu1(self.conv2(x))
        x = self.lrelu2(self.conv3(x))
        x = self.conv4(x)
        z = self.conv5(x)

        return z


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(320, 640, 5, stride=2, padding=2, output_padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.deconv2 = nn.ConvTranspose2d(640, 640, 5, stride=2, padding=2, output_padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.deconv3 = nn.ConvTranspose2d(640, 640, 5, stride=2, padding=2, output_padding=1)


    def forward(self, z_hat):
        x = self.lrelu1(self.deconv1(z_hat))
        x = self.lrelu2(self.deconv2(x))
        x = self.deconv3(x)

        return x


class SwappedDSSDFEncoder(nn.Module):
    """
    DSSDF topology adapted for the encoder / analysis transform.

    Same topology as the original SpectralSpatialFusion (DSSDF),
    but GDN is used instead of IGDN because this block is placed
    in the analysis transform.
    """

    def __init__(self, channels=320):
        super().__init__()
        mid_channels = channels // 2
        self.spectral_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, mid_channels, 1),
            GDN(mid_channels)
        )
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels
            ),
            nn.Conv2d(
                channels,
                mid_channels,
                1
            ),
            GDN(mid_channels)
        )
        self.cross_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=channels
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channels,
                channels,
                1,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 2,
                1
            ),
            SwiGLU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1
            )
        )

    def forward(self, x):
        identity = x
        spectral = self.spectral_branch(x)
        spatial = self.spatial_branch(x)
        concat = torch.cat(
            [spectral, spatial],
            dim=1
        )
        attn = self.cross_attention(concat)
        enhanced = concat * attn
        out = self.fusion(enhanced)
        out = out + identity

        return out


class SwappedESSDFDecoder(nn.Module):
    """
    ESSDF topology adapted for the decoder / synthesis transform.

    Same topology as the original SpectralSpatialDecoupling (ESSDF),
    but IGDN is used instead of GDN because this block is placed
    in the synthesis transform.
    """

    def __init__(self, channels=320):
        super().__init__()

        mid_channels = channels // 2
        self.spectral_path = nn.Sequential(
            nn.Conv2d(
                channels,
                mid_channels * 2,
                1
            ),
            SwiGLU(),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                1
            ),
            IGDN(mid_channels)
        )
        self.spatial_path = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels
            ),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels
            ),
            nn.Conv2d(
                channels,
                mid_channels,
                1
            ),
            IGDN(mid_channels)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                1
            ),
            nn.Sigmoid()
        )
        self.fusion = nn.Conv2d(
            channels,
            channels,
            1
        )

    def forward(self, x):
        identity = x
        spectral = self.spectral_path(x)
        spatial = self.spatial_path(x)
        gate = self.gate(x)
        gate_spectral = gate[
            :, :gate.size(1) // 2, :, :
        ]
        gate_spatial = gate[
            :, gate.size(1) // 2:, :, :
        ]
        weighted_spectral = (
            gate_spectral * spectral
        )
        weighted_spatial = (
            gate_spatial * spatial
        )
        out = torch.cat(
            [
                weighted_spectral,
                weighted_spatial
            ],
            dim=1
        )
        out = self.fusion(out)
        out = out + identity

        return out