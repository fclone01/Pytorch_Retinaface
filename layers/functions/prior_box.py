import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase="train"):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]  # [[16, 32], [64, 128], [256, 512]]
        self.steps = cfg["steps"]  # [8, 16, 32]
        self.clip = cfg["clip"]  # False
        self.image_size = image_size  # 640
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]  # [[80, 80], [40, 40], [20, 20]]
        self.name = "s"

    def forward(self):
        anchors = []
        # 0, [80, 80], 1, [40, 40], 2, [20, 20]
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]  # [16, 32] or [64, 128] or [256, 512]
            # 0-79, 0-79 or 0-39, 0-39 or 0-19, 0-19
            for i, j in product(range(f[0]), range(f[1])):
                # 16, 32 or 64, 128 or 256, 512
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]  # 16/640 or 32/640 or 64/640
                    s_ky = min_size / self.image_size[0]  # 16/640 or 32/640 or 64/640
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1] for x in [j + 0.5]
                    ]  # [0.5*8/640, 1.5*8/640, 2.5*8/640, ..., 79.5*8/640] or [0.5*16/640, 1.5*16/640, 2.5*16/640, ..., 39.5*16/640] or [0.5*32/640, 1.5*32/640, 2.5*32/640, ..., 19.5*32/640]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0] for y in [i + 0.5]
                    ]  # [0.5*8/640, 1.5*8/640, 2.5*8/640, ..., 79.5*8/640] or [0.5*16/640, 1.5*16/640, 2.5*16/640, ..., 39.5*16/640] or [0.5*32/640, 1.5*32/640, 2.5*32/640, ..., 19.5*32/640]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            # if>=1, then set to 1. if<=0, then set to 0
            output.clamp_(max=1, min=0)
        return output
