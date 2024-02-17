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

import argparse
import math
import random
import shutil
import sys
import time
from tqdm.rich import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

import loralib as lora

from compressai.datasets import ImageFolder
from compressai.zoo import models, load_state_dict

device = "cpu"


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    all_parameters = {n for n, p in net.named_parameters() if p.requires_grad}

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(all_parameters) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch=1,
    clip_max_norm=1.0,
):
    model.train()
    device = next(model.parameters()).device
    avg_loss = AverageMeter()
    avg_mse_loss = AverageMeter()
    avg_bpp_loss = AverageMeter()
    avg_aux_loss = AverageMeter()

    steps_to_show = 1000

    for i, d in enumerate(tqdm(train_dataloader)):
        d = d.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        avg_loss.update(out_criterion["loss"])
        avg_mse_loss.update(out_criterion["mse_loss"])
        avg_bpp_loss.update(out_criterion["bpp_loss"])
        avg_aux_loss.update(aux_loss)

        if i % steps_to_show == steps_to_show - 1:
            print(
                f"Train epoch {epoch}: ["
                f"{(i+1)*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * (i+1) / len(train_dataloader):.0f}%)]"
                f"\tLoss: {avg_loss.avg:.3f} |"
                f"\tMSE loss: {avg_mse_loss.avg * 255 ** 2 / 3:.3f} |"
                f"\tBpp loss: {avg_bpp_loss.avg:.2f} |"
                f"\tAux loss: {avg_aux_loss.avg:.2f}"
            )
            aux_loss.__init__()
            avg_mse_loss.__init__()
            avg_bpp_loss.__init__()
            avg_aux_loss.__init__()


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in tqdm(test_dataloader):
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg * 255 ** 2 / 3:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )
    return loss.avg


def save_checkpoint(state, is_best, filename, tag: str = "", epoch=0):
    filename = filename[:-8] + (f"_{tag}" if tag != "" else tag) + filename[-8:]
    best_filename = filename[:-8] + "_best" + filename[-8:]
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Train Lora",
    )
    parser.add_argument(
        "--lora_r",
        default=4,
        type=int,
        help="Lora Rank (default: %(default)s)",
    )
    parser.add_argument(
        "--hyper_lora_r",
        default=4,
        type=int,
        help="Lora Rank (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="lora_stf",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="./dataset/",
        help="Training dataset",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=10,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save model or lora to disk",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="ckpt/lora.pth.tar",
        help="Where to Save model",
    )
    parser.add_argument(
        "--seed",
        type=float,
        help="Set random seed for reproducibility",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument(
        "--pretrain_ckpt",
        required=True,
        type=str,
        help="Path to a pretrain model checkpoint",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to a lora checkpoint",
    )
    args = parser.parse_args(argv)
    return args


def load_backbone(
    arch: str, checkpoint_path: str, strict=True, device="cpu", lora_r=0, hyper_lora_r=0
) -> nn.Module:
    state_dict = load_state_dict(
        torch.load(checkpoint_path, map_location=torch.device(device))["state_dict"]
    )
    return models[arch].from_state_dict(
        state_dict, strict, lora_r=lora_r, hyper_lora_r=hyper_lora_r
    )


def fc_state_dict(net: nn.Module):
    return {n: p for n, p in net.named_parameters() if n.endswith(".quantiles")}


def NewDataLoader(args, device="cpu"):
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    return train_dataloader, test_dataloader


def NewLoraModel(args, device) -> nn.Module:
    net = load_backbone(
        args.model,
        args.pretrain_ckpt,
        strict=False,
        device=device,
        lora_r=args.lora_r,
        hyper_lora_r=args.hyper_lora_r,
    )
    lora.mark_only_lora_as_trainable(net)
    for n, p in net.named_parameters():
        if n.endswith(".quantiles"):
            p.requires_grad = True

    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    return net


def load_ckpt(args, net, device, optimizer, aux_optimizer, lr_scheduler) -> (int, int):
    print("Loading", args.ckpt)
    checkpoint = torch.load(args.ckpt, map_location=device)

    net.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    return last_epoch, best_loss


def load_lora_ckpt(
    args, net, device, optimizer, aux_optimizer, lr_scheduler
) -> (int, int):
    print("Loading", args.ckpt)
    lora_checkpoint = torch.load(args.ckpt, map_location=device)

    net.load_lora_state(lora_checkpoint["state_dict"])
    net.load_fc_state(lora_checkpoint["fc_state_dict"])
    last_epoch = lora_checkpoint["epoch"] + 1
    best_loss = lora_checkpoint["best_loss"]
    optimizer.load_state_dict(lora_checkpoint["optimizer"])
    lr_scheduler.load_state_dict(lora_checkpoint["lr_scheduler"])
    aux_optimizer.load_state_dict(lora_checkpoint["aux_optimizer"])

    return last_epoch, best_loss


def main(argv):
    args = parse_args(argv)
    print(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = NewDataLoader(args, device)
    net = NewLoraModel(args, device) if args.lora else models[args.model]()

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=2
    )
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    best_loss = float("inf")
    if args.ckpt:
        load_fn = load_lora_ckpt if args.lora else load_ckpt
        last_epoch, best_loss = load_fn(
            args, net, optimizer, aux_optimizer, lr_scheduler
        )

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if not args.save:
            continue

        if args.lora:
            ckpt_info = {
                "epoch": epoch,
                "lora_r": args.lora_r,
                "hyper_lora_r": args.hyper_lora_r,
                "state_dict": lora.lora_state_dict(net, "all"),
                "fc_state_dict": fc_state_dict(net),
                "loss": loss,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
            }
        else:
            ckpt_info = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict(),
            }

        save_checkpoint(
            ckpt_info,
            is_best,
            args.save_path,
            f"{args.lmbda}",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
