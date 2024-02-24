import os
import time
import math
from collections import defaultdict
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

import compressai
from compressai.zoo import load_state_dict, models

from tqdm import tqdm

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def load_checkpoint(arch: str, checkpoint_path: str, strict=True) -> nn.Module:
    state_dict = load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
    )
    return models[arch].from_state_dict(state_dict, strict)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, x):

    x = x.unsqueeze(0)
    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    return {
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def eval_model(model, filepaths):
    model.eval()
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for f in tqdm(filepaths):
        x = read_image(f).to(device)
        rv = inference(model, x)
        for k, v in rv.items():
            metrics[k] += v

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)

    return metrics


ckpt = "./ckpt/cnn_025.pth.tar"
net = load_checkpoint("masked_cnn", ckpt, False)

file_dir = "./dataset/test/kodim/"
filepaths = collect_images(file_dir)

compressai.set_entropy_coder("ans")

net.update(force=True)
net = net.to(device)

psnr_list = []
bpp_list = []

for i in range(32):
    net.set_mask_idx(i)
    result = eval_model(net, filepaths)
    psnr_list.append(result["psnr"])
    bpp_list.append(result["bpp"])

plt.title("Channel Influence")
plt.xlabel("Channel index")
plt.ylabel("PSNR ( dB )")
plt.bar(range(len(psnr_list)), psnr_list)
plt.savefig("channel_influence.png")
