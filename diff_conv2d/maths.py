#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DiffConv2d: Padding-free Convolution based on Preservation of
#             Differential Characteristics of Kernels
# Copyright © 2022 SciML, SCD, STFC. All rights reserved.

"""
Math related to Lagrange polynomial and conv2d indexing

Created by Kuangdai in Dec 2022
"""

from pathlib import Path

import torch


#############
# Constants #
#############

def _read_tensor_mathematica(fname):
    """ read tensor from a file saved by Mathematica """
    with open(fname) as f:
        txt = f.read().replace('{', '[').replace('}', ']')
        return torch.tensor(eval(txt), dtype=torch.float64)


# read in kernel transformation matrices
_precomputed_T_matrices = {
    ksize: _read_tensor_mathematica(
        f"{Path(__file__).parent.absolute()}/lagrange_constants/"
        f"kernel_transformation_matrix_{ksize}x{ksize}.txt")
    for ksize in [3, 5, 7]
}


def map_displacement_to_valid(H, W, gap, dtype=int, device='cpu'):
    """
    Create a map of displacements to the valid interior region
    :param H: image height
    :param W: image width
    :param gap: distance between the valid and the invalid regions
    :param dtype: data type of output
    :param device: device of output
    :return: displacement map
    """
    disp_img = torch.zeros((2, H, W), dtype=dtype, device=device)
    for disp in range(0, gap):
        disp_img[0, disp, :] = gap - disp
        disp_img[0, -disp - 1, :] = disp - gap
        disp_img[1, :, disp] = gap - disp
        disp_img[1, :, -disp - 1] = disp - gap
    return disp_img


def _cal_idx_T_mat(disp_img, W, X_int, ksize, gap, h_disp, w_disp):
    # result 1: global indices of invalid pixels
    loc_i, loc_j = torch.where((disp_img[0] == h_disp) *
                               (disp_img[1] == w_disp))
    idx_invalid = loc_i * W + loc_j

    # global index of its nearest valid center
    idx_near_center = (loc_i + h_disp) * W + loc_j + w_disp
    h_near_center = torch.div(idx_near_center, W, rounding_mode='floor')
    w_near_center = idx_near_center - h_near_center * W

    # global index of all pixels in the window
    h_near_window = h_near_center[:, None] + X_int[None, :]
    w_near_window = w_near_center[:, None] + X_int[None, :]

    # result 2: global indices of pixels in the nearest valid windows
    idx_nearest = h_near_window[:, :, None] * W + w_near_window[:, None, :]

    # kernel transformation matrices at invalid pixels
    T_mat = _precomputed_T_matrices[ksize][-h_disp + gap, -w_disp + gap]

    # return
    return idx_invalid, idx_nearest, T_mat.to(torch.ones(1).dtype)


def form_diff_system_speed(H, W, ksize):
    """
    Form a differentiation-based conv2d system (indices and matrices),
    optimized for speed
    :param H: image height
    :param W: image width
    :param ksize: kernel size
    :return: 1) global indices of invalid pixels
             2) global indices of pixels in the nearest valid windows
             3) differential kernels at invalid pixels
    """
    # create a map of displacements to valid centers
    gap = ksize // 2
    disp_img = map_displacement_to_valid(H, W, gap)

    # result 1: global indices of invalid pixels
    idx_invalid_i, idx_invalid_j = torch.where((disp_img[0] != 0) +
                                               (disp_img[1] != 0))
    idx_invalid = idx_invalid_i * W + idx_invalid_j
    N = len(idx_invalid)

    # inverse mapping from global to local
    glob_to_local = {glob.item(): local
                     for local, glob in enumerate(idx_invalid)}

    # result 2: global indices of pixels in the nearest valid windows
    idx_nearest = torch.empty((N, ksize, ksize), dtype=int)

    # result 3: kernel transformation matrices at invalid pixels
    T_mat = torch.empty((N, ksize ** 2, ksize ** 2))

    # grid coordinates
    X_int = torch.arange(-gap, gap + 1, dtype=int)

    # loop over invalid displacements to valid centers
    for h_disp in X_int:
        for w_disp in X_int:
            if h_disp == 0 and w_disp == 0:
                continue  # valid
            idx_invalid_hw, idx_nearest_hw, T_mat_hw = _cal_idx_T_mat(
                disp_img, W, X_int, ksize, gap, h_disp, w_disp)
            # local index of this pixel
            idx_this_local = [glob_to_local[glob.item()]
                              for glob in idx_invalid_hw]
            idx_nearest[idx_this_local] = idx_nearest_hw
            T_mat[idx_this_local] = T_mat_hw

    # flatten h,w for faster usage
    idx_nearest = idx_nearest.reshape(-1)
    return idx_invalid, idx_nearest, T_mat


def form_diff_system_memory(H, W, ksize):
    """
    Form a differentiation-based conv2d system (indices and matrices),
    optimized for memory
    :param H: image height
    :param W: image width
    :param ksize: kernel size
    :return: 1) global indices of invalid pixels
             2) global indices of pixels in the nearest valid windows
             3) differential kernels at invalid pixels
    """
    # create a map of displacements to valid centers
    gap = ksize // 2
    disp_img = map_displacement_to_valid(H, W, gap)

    # results as dict, keyed by (h_disp, w_disp)
    idx_invalid_dict = {}
    idx_nearest_dict = {}
    T_mat_dict = {}

    # grid coordinates
    X_int = torch.arange(-gap, gap + 1, dtype=int)

    # loop over invalid displacements to valid centers
    for h_disp in X_int:
        for w_disp in X_int:
            if h_disp == 0 and w_disp == 0:
                continue  # valid
            idx_invalid, idx_nearest, T_mat = _cal_idx_T_mat(
                disp_img, W, X_int, ksize, gap, h_disp, w_disp)
            key = f'{h_disp.item()}_{w_disp.item()}'
            idx_invalid_dict[key] = idx_invalid
            idx_nearest_dict[key] = idx_nearest.reshape(-1)
            T_mat_dict[key] = T_mat
    return idx_invalid_dict, idx_nearest_dict, T_mat_dict


if __name__ == "__main__":
    # constants
    ksize = 3
    trans_center = _precomputed_T_matrices[ksize][ksize // 2, ksize // 2]
    trans_corner = _precomputed_T_matrices[ksize][0, 0]
    # input kernel
    kernel_in = torch.ones(ksize, ksize)
    # transformed kernel
    trans_kernel_center = torch.einsum(
        'ij,j->i', trans_center, kernel_in.reshape(-1)).reshape(ksize, ksize)
    trans_kernel_corner = torch.einsum(
        'ij,j->i', trans_corner, kernel_in.reshape(-1)).reshape(ksize, ksize)
    # results
    print(kernel_in)
    print(trans_kernel_center)
    print(trans_kernel_corner)
