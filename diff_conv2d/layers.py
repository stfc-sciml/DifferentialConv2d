#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DiffConv2d: Padding-free Convolution based on Preservation of
#             Differential Characteristics of Kernels
# Copyright © 2022 SciML, SCD, STFC. All rights reserved.

"""
Classes derived from torch.nn.Conv2d based on different
boundary handling methods:

* DiffConv2dLayer: our differentiation-based method
* ExtraConv2dLayer: padding by extrapolation (Gupta and Ramani, 1978)
* RandConv2dLayer: random padding (Nguyen, 2019)
* PartConv2dLayer: partial convolution (Liu et al., 2022)
* ExplicitConv2dLayer: explicit boundary handling (Innamorati et al., 2018)

Created by Kuangdai in Dec 2022
"""

import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

from diff_conv2d.functional import DiffConv2d
from diff_conv2d.maths import map_displacement_to_valid


class DiffConv2dLayer(torch.nn.Module):
    """ class DiffConv2dLayer based on our method """

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=True,
                 # ours
                 keep_img_grad_at_invalid=True, train_edge_kernel=False,
                 optimized_for='memory'):
        """
        Constructor
        :param in_channels: same as in torch.nn.Conv2d()
        :param out_channels: same as in torch.nn.Conv2d()
        :param kernel_size: same as in torch.nn.Conv2d()
        :param groups: same as in torch.nn.Conv2d()
        :param bias: same as in torch.nn.Conv2d()
        :param keep_img_grad_at_invalid: keep image grad for backprop at
                                         invalid pixels
        :param train_edge_kernel: train an extra kernel used by invalid pixels
        :param optimized_for: code optimized for 'speed' or 'memory'
        """
        # super
        super(DiffConv2dLayer, self).__init__()
        self.inter_layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1,
            padding='same', dilation=1, groups=groups, bias=bias,
            padding_mode='zeros')

        # diff object
        self.diff_model = DiffConv2d(kernel_size)

        # edge kernel
        self.edge_layer = None
        if train_edge_kernel:
            self.edge_layer = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1,
                padding='same', dilation=1, groups=groups, bias=bias,
                padding_mode='zeros')

        # image grad
        self.keep_img_grad_at_invalid = keep_img_grad_at_invalid

        # optimized for
        self.optimized_for = optimized_for

    def forward(self, x):
        """ forward """
        # conv2d
        if self.edge_layer is None:
            z = self.diff_model.conv2d(
                x, kernel=self.inter_layer.weight,
                groups=self.inter_layer.groups,
                keep_img_grad_at_invalid=self.keep_img_grad_at_invalid,
                edge_kernel=None,
                optimized_for=self.optimized_for)
            if self.inter_layer.bias is not None:
                z += self.inter_layer.bias[None, :, None, None]
        else:
            z = self.diff_model.conv2d(
                x, kernel=self.inter_layer.weight,
                groups=self.inter_layer.groups,
                keep_img_grad_at_invalid=self.keep_img_grad_at_invalid,
                edge_kernel=self.edge_layer.weight,
                optimized_for=self.optimized_for)
            if self.inter_layer.bias is not None:
                # add self.inter_layer.bias to both interior and boundary
                z += self.inter_layer.bias[None, :, None, None]
                B, Co, _, _ = z.shape
                z_invalid = z.view(B, Co, -1)[:, :, self.diff_model.idx_invalid]
                # subtract self.inter_layer.bias from boundary
                z_invalid -= self.inter_layer.bias[None, :, None]
                # add self.edge_layer.bias to boundary
                z_invalid += self.edge_layer.bias[None, :, None]
        return z


class ExtraConv2dLayer(torch.nn.Conv2d):
    """ class ExtraConv2dLayer based on padding by extrapolation
    (Gupta and Ramani, 1978) """

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=True):
        """ constructor """
        # super
        super(ExtraConv2dLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride=1,
            padding='valid', dilation=1, groups=groups, bias=bias)

        # kernel_size must be single and odd
        assert not hasattr(kernel_size, '__len__'), \
            f'`kernel_size` must be a single `int`.'
        assert kernel_size % 2 == 1, f'`kernel_size` must be odd.'
        self.k_size = kernel_size

    def forward(self, x):
        """ forward """
        p = self.k_size // 2
        padded = F.pad(x, pad=(p, p, p, p))

        # extrapolation
        if p == 1:
            mode = 'linear'
        elif p == 2:
            mode = 'quadratic'
        elif p == 3:
            mode = 'cubic'
        else:
            assert False

        # top
        device = x.device
        padded[:, :, 0:p, :] = torch.tensor(interp1d(
            range(p, 2 * p + 1), padded[:, :, p:2 * p + 1, :].cpu().detach(),
            kind=mode, axis=2, fill_value="extrapolate")(range(0, p))).to(
            device)
        # bottom
        reverse = padded.flip(dims=(2,))
        padded[:, :, -p:, :] = torch.tensor(interp1d(
            range(p, 2 * p + 1), reverse[:, :, p:2 * p + 1, :].cpu().detach(),
            kind=mode, axis=2, fill_value="extrapolate")(
            range(p - 1, -1, -1))).to(
            device)
        # left
        padded[:, :, :, 0:p] = torch.tensor(interp1d(
            range(p, 2 * p + 1), padded[:, :, :, p:2 * p + 1].cpu().detach(),
            kind=mode, axis=3, fill_value="extrapolate")(range(0, p))).to(
            device)
        # right
        reverse = padded.flip(dims=(3,))
        padded[:, :, :, -p:] = torch.tensor(interp1d(
            range(p, 2 * p + 1), reverse[:, :, :, p:2 * p + 1].cpu().detach(),
            kind=mode, axis=3, fill_value="extrapolate")(
            range(p - 1, -1, -1))).to(
            device)
        return super(ExtraConv2dLayer, self).forward(padded)


class RandConv2dLayer(torch.nn.Conv2d):
    """ class RandConv2dLayer based on random padding (Nguyen, 2019) """

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=True):
        """ constructor """
        # super
        super(RandConv2dLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride=1,
            padding='valid', dilation=1, groups=groups, bias=bias)

        # kernel_size must be single and odd
        assert not hasattr(kernel_size, '__len__'), \
            f'`kernel_size` must be a single `int`.'
        assert kernel_size % 2 == 1, f'`kernel_size` must be odd.'
        self.k_size = kernel_size

    def forward(self, x):
        """ forward """
        p = self.k_size // 2
        padded = F.pad(x, pad=(p, p, p, p))

        N, C, H, W = x.shape
        # top
        value = x[:, :, 0:p + 1, :]
        mean = value.mean(dim=[2, 3])
        std = value.std(dim=[2, 3])
        pad = torch.randn((p, W + 2 * p), device=x.device)
        padded[:, :, 0:p, :] += \
            pad[None, None, :, :] * std[:, :, None, None] + \
            mean[:, :, None, None]

        # bottom
        value = x[:, :, -(p + 1):, :]
        mean = value.mean(dim=[2, 3])
        std = value.std(dim=[2, 3])
        pad = torch.randn((p, W + 2 * p), device=x.device)
        padded[:, :, -p:, :] += \
            pad[None, None, :, :] * std[:, :, None, None] + \
            mean[:, :, None, None]

        # left
        value = x[:, :, :, 0:p + 1]
        mean = value.mean(dim=[2, 3])
        std = value.std(dim=[2, 3])
        pad = torch.randn((H + 2 * p, p), device=x.device)
        padded[:, :, :, 0:p] += \
            pad[None, None, :, :] * std[:, :, None, None] + \
            mean[:, :, None, None]

        # right
        value = x[:, :, :, -(p + 1):]
        mean = value.mean(dim=[2, 3])
        std = value.std(dim=[2, 3])
        pad = torch.randn((H + 2 * p, p), device=x.device)
        padded[:, :, :, -p:] += \
            pad[None, None, :, :] * std[:, :, None, None] + \
            mean[:, :, None, None]

        # four corners added twice
        padded[:, :, 0:p, 0:p] /= 2
        padded[:, :, 0:p, -p:] /= 2
        padded[:, :, -p:, 0:p] /= 2
        padded[:, :, -p:, -p:] /= 2
        return super(RandConv2dLayer, self).forward(padded)


class PartConv2dLayer(torch.nn.Conv2d):
    """ class PartConv2dLayer based on partial conv (Liu et al., 2022) """

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=True):
        """ constructor """
        # super
        super(PartConv2dLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride=1,
            padding='same', dilation=1, groups=groups, bias=bias,
            padding_mode='zeros')

        # kernel_size must be single and odd
        assert not hasattr(kernel_size, '__len__'), \
            f'`kernel_size` must be a single `int`.'
        assert kernel_size % 2 == 1, f'`kernel_size` must be odd.'
        self.k_size = kernel_size

    def forward(self, img):
        """ forward """
        # compute r matrix
        N, Ci, H, W = img.shape
        gap = self.k_size // 2
        disp_img = map_displacement_to_valid(H, W, gap, dtype=img.dtype,
                                             device=img.device)
        dist_h, dist_w = disp_img[0].abs(), disp_img[1].abs()
        n_invalid = (dist_h + dist_w) * self.k_size - dist_h * dist_w
        r = (self.k_size ** 2) / (self.k_size ** 2 - n_invalid)

        # convolve
        z = super(PartConv2dLayer, self).forward(img)

        # scale with r
        if self.bias is not None:
            z -= self.bias[None, :, None, None]
            z *= r[None, None, :, :]
            z += self.bias[None, :, None, None]
        else:
            z *= r
        return z


class ExplicitConv2dLayer(torch.nn.Module):
    """ class ExplicitConv2dLayer based on explicit boundary handling
    (Innamorati et al., 2018) """

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, bias=True):
        """ constructor """
        super(ExplicitConv2dLayer, self).__init__()

        # kernel_size must be single and odd
        assert not hasattr(kernel_size, '__len__'), \
            f'`kernel_size` must be a single `int`.'
        assert kernel_size % 2 == 1, f'`kernel_size` must be odd.'
        self.k_size = kernel_size
        self.groups = groups

        # interior must come first for seed preservation
        interior_layer = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1,
            padding='same', dilation=1, groups=groups, bias=bias,
            padding_mode='zeros')

        # create copies of kernels (e.g., 9 copies for 3x3),
        # keyed by displacements to valid regions
        gap = self.k_size // 2
        self.layers_by_disp = torch.nn.ModuleDict()
        invalid_disps = range(-gap, gap + 1)
        for h_disp in invalid_disps:
            for w_disp in invalid_disps:
                key = f'{h_disp}__{w_disp}'
                if h_disp == w_disp:
                    # interior
                    self.layers_by_disp[key] = interior_layer
                else:
                    # boundary
                    self.layers_by_disp[key] = torch.nn.Conv2d(
                        in_channels, out_channels, kernel_size, stride=1,
                        padding='same', dilation=1, groups=groups, bias=bias,
                        padding_mode='zeros')

    def forward(self, img):
        """ forward """
        # convolve with all copies of kernels
        z_by_disp = {}
        for key, layer in self.layers_by_disp.items():
            z_by_disp[key] = F.conv2d(img, layer.weight, padding='same',
                                      groups=self.groups)
            if layer.bias is not None:
                z_by_disp[key] += layer.bias[None, :, None, None]

        # displacement map
        _, _, H, W = img.shape
        gap = self.k_size // 2
        disp_img = map_displacement_to_valid(H, W, gap)

        # choose the right convolved result for each pixel
        z = torch.zeros_like(list(z_by_disp.values())[0])
        for key, z_value in z_by_disp.items():
            h_disp, w_disp = [int(d) for d in key.split('__')]
            loc_i, loc_j = torch.where((disp_img[0] == h_disp) *
                                       (disp_img[1] == w_disp))
            z[:, :, loc_i, loc_j] += z_value[:, :, loc_i, loc_j]
        return z


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    batch_size, ch_in, ch_out, H_, W_, k_size = 4, 4, 8, 500, 300, 5
    model_s = DiffConv2dLayer(ch_in, ch_out, k_size, groups=2,
                              optimized_for='speed')
    x_ = torch.rand((batch_size, ch_in, H_, W_))
    z_s = model_s.forward(x_)
    model_s.optimized_for = 'memory'
    z_m = model_s.forward(x_)
    print(z_s.shape)
    print((z_s - z_m).abs().max())
