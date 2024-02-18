# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import loralib as lora
from .win_attention import WinBasedAttention, WinBasedAttentionLora

__all__ = [
    "conv3x3",
    "subpel_conv3x3",
    "conv1x1",
    "lora_conv3x3",
    "lora_subpel_conv3x3",
    "lora_conv1x1",
    "Win_noShift_Attention",
    "Win_noShift_Attention_Lora",
]


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch: int, out_ch: int, r: int = 1) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def lora_conv1x1(
    in_ch: int, out_ch: int, stride: int = 1, lora_r: int = 0, merge_weights=True
) -> nn.Module:
    """1x1 convolution."""
    return lora.Conv2d(
        in_ch,
        out_ch,
        kernel_size=1,
        stride=stride,
        r=lora_r,
        merge_weights=merge_weights,
    )


def lora_conv3x3(
    in_ch: int, out_ch: int, stride: int = 1, lora_r: int = 0, merge_weights=True
) -> nn.Module:
    """3x3 convolution with padding."""
    return lora.Conv2d(
        in_ch,
        out_ch,
        kernel_size=3,
        stride=stride,
        padding=1,
        r=lora_r,
        merge_weights=merge_weights,
    )


def lora_subpel_conv3x3(
    in_ch: int, out_ch: int, r: int = 1, lora_r: int = 0, merge_weights=True
) -> nn.Sequential:
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        lora.Conv2d(
            in_ch,
            out_ch * r**2,
            kernel_size=3,
            padding=1,
            r=lora_r,
            merge_weights=merge_weights,
        ),
        nn.PixelShuffle(r),
    )


class Win_noShift_Attention(nn.Module):
    """Window-based self-attention module."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.GELU(),
                    conv3x3(N // 2, N // 2),
                    nn.GELU(),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
            ),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class Win_noShift_Attention_Lora(Win_noShift_Attention):

    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=8,
        shift_size=0,
        lora_r=0,
        merge_weights=True,
        enable_lora=[True, True, True],
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
        )
        N = dim

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    lora_conv1x1(N, N // 2, lora_r=lora_r, merge_weights=merge_weights),
                    nn.GELU(),
                    lora_conv3x3(
                        N // 2, N // 2, lora_r=lora_r, merge_weights=merge_weights
                    ),
                    nn.GELU(),
                    lora_conv1x1(N // 2, N, lora_r=lora_r, merge_weights=merge_weights),
                )
                self.relu = nn.GELU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            WinBasedAttentionLora(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                lora_r=lora_r,
                merge_weights=merge_weights,
                enable_lora=enable_lora,
            ),
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            lora_conv1x1(N, N, lora_r=lora_r, merge_weights=merge_weights),
        )
