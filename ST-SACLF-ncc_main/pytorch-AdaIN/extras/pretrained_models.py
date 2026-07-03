import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.utils as tutils

#import utils
#from transformer_net import TransformerNet
#from vgg import Vgg16

from torch import nn
import torch.nn.functional as F

from collections import namedtuple


from torchvision import models
from torchvision.models import VGG16_Weights
from .model_blocks import *

import torch
from PIL import Image

# ST-SACLF-ncc_main/models/vgg_normalised.pth（相对本文件：extras -> pytorch-AdaIN -> ST-SACLF-ncc_main）
_ST_SACLF_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGG_NORMALISED = _ST_SACLF_ROOT / "models" / "vgg_normalised.pth"


def resolve_vgg_normalised_path(weights_path: Optional[str] = None) -> Path:
    """解析 AdaIN VGG 权重路径；默认 ST-SACLF-ncc_main/models/vgg_normalised.pth。"""
    candidates = [
        Path(weights_path) if weights_path else None,
        Path(os.environ["VGG_NORMALISED_PATH"])
        if os.environ.get("VGG_NORMALISED_PATH")
        else None,
        DEFAULT_VGG_NORMALISED,
        Path(__file__).resolve().parents[1] / "models" / "vgg_normalised.pth",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError(
        "未找到 vgg_normalised.pth，请将权重放在 "
        f"{DEFAULT_VGG_NORMALISED} 或通过 --vgg / VGG_NORMALISED_PATH 指定。"
    )


def get_project_in(model: torch.nn.Sequential):
    out_dim = 0
    for layer in model.children():
        if 'Conv2d' in str(type(layer)):
            out_dim = layer.out_channels
    return out_dim


#the pretrained model
class Vgg16(torch.nn.Module):
    def __init__(self, output_layers, requires_grad=False, weights_path: Optional[str] = None):
        super(Vgg16, self).__init__()
        vgg = models.vgg19().features
        vgg_pretrained_features = vgg

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        reflect = nn.ReflectionPad2d((1, 1, 1, 1))

        dense = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.slice5 = torch.nn.Sequential()#global features
        # old_pretrained_state_space = torch.load('models/vgg_normalised.pth')
        # temp_state_space = {'0.weight': old_pretrained_state_space['0.weight'], '0.bias': old_pretrained_state_space['0.bias']}
        extra = torch.nn.Conv2d(3, 3, (1, 1))
        self.slice1.add_module(str(0), extra)
        self.slice1.add_module(str(1), reflect)
        # self.slice1.load_state_dict(temp_state_space)
        for x in range(2, output_layers[0]+1+2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x-2])
        for x in range(output_layers[0]+1, output_layers[1]+1):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[1]+1, output_layers[2]+1):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[2]+1, output_layers[3]+1):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[3]+1, len(vgg_pretrained_features)-1):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.slice5.add_module(str(len(vgg_pretrained_features) - 1), dense)

        vgg_ckpt = resolve_vgg_normalised_path(weights_path)
        pretrained_state_space = torch.load(str(vgg_ckpt), map_location="cpu")
        temp_state_space = self.state_dict()
        for k_p, k_n in zip(list(pretrained_state_space.keys()), list(temp_state_space.keys())):
            temp_state_space[k_n] = pretrained_state_space[k_p]
        self.load_state_dict(temp_state_space)

        self.project_ins = [get_project_in(self.slice1),]
        self.project_ins.append(get_project_in(self.slice2))
        self.project_ins.append(get_project_in(self.slice3))
        self.project_ins.append(get_project_in(self.slice4))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        l0 = h
        h = self.slice2(h)
        l1 = h
        h = self.slice3(h)
        l2 = h
        h = self.slice4(h)
        l3 = h

        g = self.slice5(h)

        vgg_outputs = namedtuple("VggOutputs", ['style_lyr1', 'style_lyr2', 'style_lyr3', 'content_lyr'])
        out = vgg_outputs(l0, l1, l2, l3)

        return out, g


# class TransformerDecoder(nn.Module):
#


decoder = nn.Sequential(
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

class Vgg16Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
            #nn.ReLU(),
        )

        self.net = nn.ModuleList([block1, block2, block3, block4, block5])

    def forward(self, x):
        for ix, module in enumerate(self.net):
            x = module(x)
            #print(x.shape)
            if ix < len(self.net) -1:#at max pool locations
                x = F.interpolate(x, scale_factor= 2, mode='nearest')
        return x
#VGG-13


#VGG-19


#ResNet-34


#ResNet-50


if __name__ == '__main__':
    temp_img = torch.randn((8,3,255,255))
    temp = Vgg16([2, 7, 12, 19])
    print(temp)
    #print(temp(temp_img)[1].shape)
