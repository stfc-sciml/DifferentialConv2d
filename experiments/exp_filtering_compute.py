from pathlib import Path

import numpy as np
import torch
from scipy.special import chebyu, sph_harm
from tqdm import tqdm

from exp_filtering_methods import methods_dict, conv2d_padding


def f_chebyu(n):
    """ Chebyshev """
    length = 500
    h_crd = np.linspace(.3, .8, length)
    w_crd = np.linspace(.3, .8, length)
    h, w = np.meshgrid(h_crd, w_crd, indexing='ij')
    un = chebyu(n)
    return un(h) * un(w) * np.sin((h + w) * n)


def f_sph(m, n):
    """ spherical harmonics """
    length = 500
    h_crd = np.linspace(.3, .8, length)
    w_crd = np.linspace(.3, .8, length)
    h_crd = (h_crd - h_crd.min()) / (
            h_crd.max() - h_crd.min()) * np.pi * .2 + np.pi * .4
    w_crd = (w_crd - w_crd.min()) / (
            w_crd.max() - w_crd.min()) * np.pi * .2 / 6 * 5 + (
                    np.pi * .5 - np.pi * .2 / 6 * 5 / 2)
    h, w = np.meshgrid(h_crd, w_crd, indexing='ij')
    return np.real(sph_harm(m, n, h, w)) * np.sin((h + w) * n)


def f_NS(batches, times):
    """ NS solution """
    data = np.load('exp_data/NS/NS_Re500_s256_T100_test.npy')
    return data[batches, times]


def l1_error(images, nk, ksize):
    """ l1 error """
    l1 = torch.zeros(len(images), len(methods_dict))
    for i_image, image in enumerate(tqdm(images)):
        image = torch.from_numpy(image).to(torch.float) \
            .unsqueeze(0).unsqueeze(0)
        # random kernel
        kernel = torch.randn((nk, 1, ksize, ksize))
        # ground truth
        z_true = conv2d_padding(image, kernel, ignore_padding=True)
        # crop image by one pixel
        p = ksize // 2
        image_crop = image[:, :, p:-p, p:-p]
        _, _, H, W = image_crop.shape
        n_inv = H * W - (H - p * 2) * (W - p * 2)
        # pred
        for i_method, (name, method) in enumerate(methods_dict.items()):
            z_pred = method(image_crop, kernel)
            l1[i_image, i_method] = \
                (z_pred - z_true).abs().sum() / nk / n_inv
    return l1


if __name__ == '__main__':
    n_kernels = 100
    kernel_size = 3
    out_dir = Path('./exp_results/filtering')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chebyshev
    print('Computing for Chebyshev')
    orders = np.arange(0, 101, 10)
    orders[0] = 1
    x = [f_chebyu(order) for order in orders]
    err = l1_error(x, n_kernels, kernel_size)
    torch.save({'crd': orders, 'err': err}, out_dir / 'chebyu.pt')

    # spherical harmonics
    print('Computing for spherical harmonics')
    ms_sph = np.arange(5, 56, 5)
    x = [f_sph(m_sph, m_sph * 2) for m_sph in ms_sph]
    err = l1_error(x, n_kernels, kernel_size)
    torch.save({'crd': ms_sph, 'err': err}, out_dir / 'sph.pt')

    # NS solution
    print('Computing for NS solution')
    bh = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    x = f_NS(bh - 1, 0)
    err = l1_error(x, n_kernels, kernel_size)
    bh[0] = 0
    torch.save({'crd': bh, 'err': err}, out_dir / 'NS.pt')
