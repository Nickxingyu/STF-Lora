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
from torch.nn import functional as F

import loralib as lora

from typing import Optional, List


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key: str,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",  # 修改了
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def lora_conv(
    in_channels, out_channels, kernel_size=5, stride=2, lora_r=0, merge_weights=True
):
    return lora.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        r=lora_r,
        merge_weights=merge_weights,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):  # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class ConvTranspose2dLora(lora.ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(ConvTranspose2dLora, self).__init__(nn.ConvTranspose2d, *args, **kwargs)

    def forward(self, x, output_size: Optional[List[int]] = None):
        if self.r > 0 and not self.merged:
            if self.conv.padding_mode != "zeros":
                raise ValueError(
                    "Only `zeros` padding mode is supported for ConvTranspose2d"
                )

            assert isinstance(self.conv.padding, tuple)
            # One cannot replace List by Tuple or Sequence in "_output_padding" because
            # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
            num_spatial_dims = 2
            output_padding = self.conv._output_padding(
                x,
                output_size,
                self.conv.stride,
                self.conv.padding,
                self.conv.kernel_size,  # type: ignore[arg-type]
                num_spatial_dims,
                self.conv.dilation,
            )  # type: ignore[arg-type]

            return F.conv_transpose2d(
                x,
                self.conv.weight
                + (self.lora_B @ self.lora_A).view(self.conv.weight.shape)
                * self.scaling,
                self.conv.bias,
                self.conv.stride,
                self.conv.padding,
                output_padding,
                self.conv.groups,
                self.conv.dilation,
            )

        return self.conv(x)


def lora_deconv(
    in_channels, out_channels, kernel_size=5, stride=2, lora_r=0, merge_weights=True
):
    return ConvTranspose2dLora(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
        r=lora_r,
        merge_weights=merge_weights,
    )
