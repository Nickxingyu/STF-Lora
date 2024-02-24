import math
import torch
import torch.nn as nn

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from .utils import conv, deconv, update_registered_buffers, lora_conv, lora_deconv
from compressai.ops import ste_round
from compressai.layers import (
    conv3x3,
    subpel_conv3x3,
    Win_noShift_Attention,
    Win_noShift_Attention_Lora,
    lora_conv3x3,
    lora_subpel_conv3x3,
)
from .base import CompressionModel

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class WACNN(CompressionModel):
    """CNN based model"""

    def __init__(self, N=192, M=320, **kwargs):
        super().__init__(**kwargs)
        self.num_slices = 10
        self.max_support_slices = 5

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            )
            for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            )
            for i in range(10)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(320 + 32 * min(i + 1, 6), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 176, stride=1, kernel_size=3),
                nn.GELU(),
                conv(176, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, 64, stride=1, kernel_size=3),
                nn.GELU(),
                conv(64, 32, stride=1, kernel_size=3),
            )
            for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict, strict=True, update=True):
        if update:
            update_registered_buffers(
                self.gaussian_conditional,
                "gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )
        super().load_state_dict(state_dict, strict, update)

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["g_a.0.weight"].size(0)
        # M = state_dict["g_a.6.weight"].size(0)
        # net = cls(N, M)
        net = cls(192, 320)
        net.load_state_dict(state_dict, strict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood

    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class WACNNWithLora(WACNN):
    """CNN based model"""

    def __init__(
        self,
        N=192,
        M=320,
        lora_r=8,
        hyper_lora_r=8,
        merge_weights=True,
        enable_lora=[True, True, True],
        **kwargs,
    ):
        super().__init__(N, M, **kwargs)

        self.g_a = nn.Sequential(
            lora_conv(
                3,
                N,
                kernel_size=5,
                stride=2,
                lora_r=2,
                merge_weights=merge_weights,
            ),
            GDN(N),
            lora_conv(
                N,
                N,
                kernel_size=5,
                stride=2,
                lora_r=lora_r,
                merge_weights=merge_weights,
            ),
            GDN(N),
            Win_noShift_Attention_Lora(
                dim=N,
                num_heads=8,
                window_size=8,
                shift_size=4,
                lora_r=lora_r,
                merge_weights=merge_weights,
                enable_lora=enable_lora,
            ),
            lora_conv(
                N,
                N,
                kernel_size=5,
                stride=2,
                lora_r=lora_r,
                merge_weights=merge_weights,
            ),
            GDN(N),
            lora_conv(
                N,
                M,
                kernel_size=5,
                stride=2,
                lora_r=4,
                merge_weights=merge_weights,
            ),
            Win_noShift_Attention_Lora(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                lora_r=4,
                merge_weights=merge_weights,
                enable_lora=enable_lora,
            ),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention_Lora(
                dim=M,
                num_heads=8,
                window_size=4,
                shift_size=2,
                lora_r=4,
                merge_weights=merge_weights,
                enable_lora=enable_lora,
            ),
            lora_deconv(
                M,
                N,
                kernel_size=5,
                stride=2,
                lora_r=4,
                merge_weights=merge_weights,
            ),
            GDN(N, inverse=True),
            lora_deconv(
                N,
                N,
                kernel_size=5,
                stride=2,
                lora_r=lora_r,
                merge_weights=merge_weights,
            ),
            GDN(N, inverse=True),
            Win_noShift_Attention_Lora(
                dim=N,
                num_heads=8,
                window_size=8,
                shift_size=4,
                lora_r=lora_r,
                merge_weights=merge_weights,
                enable_lora=enable_lora,
            ),
            lora_deconv(
                N,
                N,
                kernel_size=5,
                stride=2,
                lora_r=lora_r,
                merge_weights=merge_weights,
            ),
            GDN(N, inverse=True),
            lora_deconv(
                N,
                3,
                kernel_size=5,
                stride=2,
                lora_r=2,
                merge_weights=merge_weights,
            ),
        )

        self.h_a = nn.Sequential(
            lora_conv3x3(320, 320, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_conv3x3(320, 288, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_conv3x3(
                288, 256, lora_r=hyper_lora_r, merge_weights=merge_weights, stride=2
            ),
            nn.GELU(),
            lora_conv3x3(256, 224, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_conv3x3(
                224, 192, lora_r=hyper_lora_r, merge_weights=merge_weights, stride=2
            ),
        )

        self.h_mean_s = nn.Sequential(
            lora_conv3x3(192, 192, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_subpel_conv3x3(
                192, 224, 2, lora_r=hyper_lora_r, merge_weights=merge_weights
            ),
            nn.GELU(),
            lora_conv3x3(224, 256, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_subpel_conv3x3(
                256, 288, 2, lora_r=hyper_lora_r, merge_weights=merge_weights
            ),
            nn.GELU(),
            lora_conv3x3(288, 320, lora_r=hyper_lora_r, merge_weights=merge_weights),
        )

        self.h_scale_s = nn.Sequential(
            lora_conv3x3(192, 192, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_subpel_conv3x3(
                192, 224, 2, lora_r=hyper_lora_r, merge_weights=merge_weights
            ),
            nn.GELU(),
            lora_conv3x3(224, 256, lora_r=hyper_lora_r, merge_weights=merge_weights),
            nn.GELU(),
            lora_subpel_conv3x3(
                256, 288, 2, lora_r=hyper_lora_r, merge_weights=merge_weights
            ),
            nn.GELU(),
            lora_conv3x3(288, 320, lora_r=hyper_lora_r, merge_weights=merge_weights),
        )
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                lora_conv(
                    320 + 32 * min(i, 5),
                    224,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    224,
                    176,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    176,
                    128,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    128,
                    64,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    64,
                    32,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
            )
            for i in range(10)
        )
        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                lora_conv(
                    320 + 32 * min(i, 5),
                    224,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    224,
                    176,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    176,
                    128,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    128,
                    64,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    64,
                    32,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
            )
            for i in range(10)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                lora_conv(
                    320 + 32 * min(i + 1, 6),
                    224,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    224,
                    176,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    176,
                    128,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    128,
                    64,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
                nn.GELU(),
                lora_conv(
                    64,
                    32,
                    stride=1,
                    kernel_size=3,
                    lora_r=hyper_lora_r,
                    merge_weights=merge_weights,
                ),
            )
            for i in range(10)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    @classmethod
    def from_state_dict(cls, state_dict, strict: bool = True, lora_r=0, hyper_lora_r=0):
        net = cls(192, 320, lora_r=lora_r, hyper_lora_r=hyper_lora_r)
        net.load_state_dict(state_dict, strict)
        return net

    def load_lora_state(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict=strict, update=False)

    def load_fc_state(self, state_dict, strict=False):
        super().load_state_dict(state_dict, strict, update=False)


class ChannelMask(nn.Module):
    def __init__(
        self, num_channels: int = 192, target_idx: int = -1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_idx = target_idx
        self.num_channels = num_channels

    def set_target_idx(self, idx: int):
        self.target_idx = idx

    def forward(self, x):
        mask = torch.ones_like(x)
        _, C, _, _ = mask.shape
        if self.target_idx >= C or self.target_idx < 0:
            return x

        mask[:, self.target_idx] = torch.zeros_like(mask[:, self.target_idx])
        return x * mask


class MaskedWACNN(WACNN):
    def __init__(self, N=192, M=320, target_idx=-1, **kwargs):
        super().__init__(N, M, **kwargs)
        self.channel_mask = ChannelMask(320, target_idx)
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
        )
        self.g_s = nn.Sequential(
            Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            self.channel_mask,
            conv3x3(320, 320),
            nn.GELU(),
            conv3x3(320, 288),
            nn.GELU(),
            conv3x3(288, 256, stride=2),
            nn.GELU(),
            conv3x3(256, 224),
            nn.GELU(),
            conv3x3(224, 192, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

        self.h_scale_s = nn.Sequential(
            conv3x3(192, 192),
            nn.GELU(),
            subpel_conv3x3(192, 224, 2),
            nn.GELU(),
            conv3x3(224, 256),
            nn.GELU(),
            subpel_conv3x3(256, 288, 2),
            nn.GELU(),
            conv3x3(288, 320),
        )

    def set_mask_idx(self, idx: int):
        self.channel_mask.set_target_idx(idx)

    def get_num_channels(self):
        return self.channel_mask.num_channels
