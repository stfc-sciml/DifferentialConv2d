#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DiffConv2d: Padding-free Convolution based on Preservation of
#             Differential Characteristics of Kernels
# Copyright © 2022 SciML, SCD, STFC. All rights reserved.

"""
Class DiffConv2d that implements our differentiation-based convolution.
Usage is similar to `torch.nn.functional.conv2d()`.

Created by Kuangdai in Dec 2022
"""

import torch
import torch.nn.functional as F

from diff_conv2d.maths import form_diff_system_speed, form_diff_system_memory


class DiffConv2d:
    """ class DiffConv2d """

    def __init__(self, kernel_size):
        """
        Constructor
        :param kernel_size: kernel size
        """
        self.kernel_size = kernel_size

        # kernel_size must be single and in [3, 5, 7]
        assert not hasattr(kernel_size, '__len__'), \
            f'`kernel_size` must be a single `int`.'
        assert kernel_size in [3, 5, 7], f'`kernel_size` must be 3, 5 or 7.'

        # diff system
        self.image_size = (-1, -1)
        self.idx_invalid, self.idx_nearest, self.T_mat = None, None, None
        self.optimized_for = None

    def _check_sizes(self, img, kernel, groups):
        """ check sizes """
        B, Ci, H, W = img.shape
        Co, Ci_d_G, h, w = kernel.shape
        G, Co_d_G = groups, Co // groups
        n, p = self.kernel_size, self.kernel_size // 2
        assert (Ci_d_G == Ci // G and Ci % G == 0 and Co % G == 0), \
            'Incompatible `groups` and channel numbers.'
        assert h == w and h == n, 'Incompatible kernel size.'
        return B, G, Ci, Co, Ci_d_G, Co_d_G, H, W, h, w, n, p

    def conv2d(self, img, kernel, groups=1,
               keep_img_grad_at_invalid=True, edge_kernel=None,
               optimized_for='memory'):
        """
        Perform conv2d()
        :param img: input images
        :param kernel: input kernels
        :param groups: number of groups
        :param keep_img_grad_at_invalid: keep image grad for backprop at
                                         invalid pixels
        :param edge_kernel: an extra kernel used by invalid pixels
        :param optimized_for: code optimized for 'speed' or 'memory'
        :return: output images
        """
        # STEP 0: check size
        B, G, Ci, Co, Ci_d_G, Co_d_G, H, W, h, w, n, p = \
            self._check_sizes(img, kernel, groups)

        # STEP 1: torch.nn.functional.conv2d() for valid pixels
        # this padding only for getting the right shape
        R = F.conv2d(img, kernel, padding=p, groups=groups)

        # STEP 2: form diff system
        if self.image_size != (H, W) or self.optimized_for != optimized_for:
            assert optimized_for in ['speed', 'memory'], \
                "`optimized_for` must be 'speed' or 'memory'."
            if optimized_for == 'speed':
                self.idx_invalid, self.idx_nearest, self.T_mat = \
                    form_diff_system_speed(H, W, ksize=n)
            else:
                self.idx_invalid, self.idx_nearest, self.T_mat = \
                    form_diff_system_memory(H, W, ksize=n)
            self.image_size = (H, W)
            self.optimized_for = optimized_for

        def _shared_process(idx_invalid, idx_nearest, T_mat):
            n_invalid, n2 = len(idx_invalid), n ** 2
            if optimized_for == 'speed':
                trans_kernel_str = 'NKj,OIj->OINK'
                trans_kernel_dim = (G, Co_d_G, Ci_d_G, n_invalid, n2)
                R_invalid_str = 'GOiNm,BGiNm->BGON'
            else:
                trans_kernel_str = 'Kj,OIj->OIK'
                trans_kernel_dim = (G, Co_d_G, Ci_d_G, n2)
                R_invalid_str = 'GOim,BGiNm->BGON'

            # STEP 3: compute transformed kernel
            kernel_use = kernel if edge_kernel is None else edge_kernel
            trans_kernel = torch.einsum(trans_kernel_str, T_mat,
                                        kernel_use.reshape(Co, Ci_d_G, n2))

            # add group dim
            trans_kernel = trans_kernel.reshape(trans_kernel_dim)

            # STEP 4: get values from nearest windows
            # get nearest
            U_nearest = img.reshape(B, Ci, -1)[:, :, idx_nearest]
            # the following reshape includes two reshapes:
            # (B, Ci, -1) => (B, Ci, n_invalid, n2)
            # (B, Ci, n_invalid, n2) => (B, G, Ci_d_G, n_invalid, n2)
            U_nearest = U_nearest.reshape(B, G, Ci_d_G, n_invalid, n2)
            # detach image grad if not keeping
            if not keep_img_grad_at_invalid:
                U_nearest = U_nearest.detach()

            # STEP 5: convolution
            R_invalid = torch.einsum(R_invalid_str, trans_kernel, U_nearest)
            # merge group dim
            R_invalid = R_invalid.reshape(B, Co, n_invalid)
            # final replacement
            R.view(B, Co, -1)[:, :, idx_invalid] = R_invalid[:, :, :]

        if optimized_for == 'speed':
            self.idx_invalid = self.idx_invalid.to(img.device)
            self.idx_nearest = self.idx_nearest.to(img.device)
            self.T_mat = self.T_mat.to(img.device)
            _shared_process(self.idx_invalid, self.idx_nearest, self.T_mat)
        else:
            for key in self.idx_invalid.keys():
                self.idx_invalid[key] = self.idx_invalid[key].to(img.device)
                self.idx_nearest[key] = self.idx_nearest[key].to(img.device)
                self.T_mat[key] = self.T_mat[key].to(img.device)
                _shared_process(self.idx_invalid[key], self.idx_nearest[key],
                                self.T_mat[key])
        return R


if __name__ == "__main__":
    # sizes
    torch.set_default_dtype(torch.float64)
    B_, Ci_, H_, W_ = 8, 4, 500, 300
    n_ = 5
    p_ = n_ // 2
    Co_, groups_ = 1, 1
    img_ = torch.rand(B_, Ci_, H_, W_)
    kernel_ = torch.rand(Co_, Ci_ // groups_, n_, n_)
    model = DiffConv2d(kernel_size=n_)
    res_s = model.conv2d(img_, kernel_, groups=groups_, optimized_for='speed')
    res_m = model.conv2d(img_, kernel_, groups=groups_, optimized_for='memory')
    print(res_s.shape)
    print((res_s - res_m).abs().max())
