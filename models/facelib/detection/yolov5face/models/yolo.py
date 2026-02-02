import math
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch import nn

from models.facelib.detection.yolov5face.models.common import (
    C3,
    NMS,
    SPP,
    AutoShape,
    Bottleneck,
    BottleneckCSP,
    Concat,
    Conv,
    DWConv,
    Focus,
    ShuffleV2Block,
    StemBlock,
)
from models.facelib.detection.yolov5face.models.experimental import CrossConv, MixConv2d
from models.facelib.detection.yolov5face.utils.autoanchor import check_anchor_order
from models.facelib.detection.yolov5face.utils.general import make_divisible
from models.facelib.detection.yolov5face.utils.torch_utils import copy_attr, fuse_conv_and_bn


class Detect(nn.Module):
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5 + 10

        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        if self.export:
            for i in range(self.nl):
                x[i] = self.m[i](x[i])
            return x
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = torch.full_like(x[i], 0)
                y[..., [0, 1, 2, 3, 4, 15]] = x[i][..., [0, 1, 2, 3, 4, 15]].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]

                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]

                y[..., 5:7] = (
                    y[..., 5:7] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                )
                y[..., 7:9] = (
                    y[..., 7:9] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                )
                y[..., 9:11] = (
                    y[..., 9:11] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                )
                y[..., 11:13] = (
                    y[..., 11:13] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                )
                y[..., 13:15] = (
                    y[..., 13:15] * self.anchor_grid[i] + self.grid[i].to(x[i].device) * self.stride[i]
                )

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None):
        super().__init__()
        self.yaml_file = Path(cfg).name
        with Path(cfg).open(encoding="utf8") as f:
            self.yaml = yaml.safe_load(f)


        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]


        m = self.model[-1]
        if isinstance(m, Detect):
            s = 128
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()

    def forward(self, x):
        return self.forward_once(x)

    def forward_once(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            x = m(x)
            y.append(x if m.i in self.save else None)

        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for mi in m.m:
            b = mi.bias.detach().view(m.na, -1).T
            print(("%6g Conv2d.bias:" + "%10.3g" * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):
        print("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, Conv) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None
        return self

    def nms(self, mode=True):
        present = isinstance(self.model[-1], NMS)
        if mode and not present:
            print("Adding NMS... ")
            m = NMS()
            m.f = -1
            m.i = self.model[-1].i + 1
            self.model.add_module(name=str(m.i), module=m)
            self.eval()
        elif not mode and present:
            print("Removing NMS... ")
            self.model = self.model[:-1]
        return self

    def autoshape(self):
        print("Adding autoShape... ")
        m = AutoShape(self)
        copy_attr(m, self, include=("yaml", "nc", "hyp", "names", "stride"), exclude=())
        return m


def parse_model(d, ch):
    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [
            Conv,
            Bottleneck,
            SPP,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            ShuffleV2Block,
            StemBlock,
        ]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
