import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

from diff_conv2d.functional import DiffConv2d
from diff_conv2d.maths import map_displacement_to_valid


def conv2d_padding(image, kernel, padding_mode='constant',
                   ignore_padding=False):
    """ padding-based conv2d methods """
    if ignore_padding:
        return F.conv2d(image, kernel)

    k_size = kernel.shape[2]
    p = k_size // 2
    padded = F.pad(image, pad=(p, p, p, p), mode=padding_mode)
    return F.conv2d(padded, kernel)


def conv2d_part(image, kernel):
    """ partial convolution """
    # compute r matrix
    N, Ci, H, W = image.shape
    k_size = kernel.shape[2]
    p = k_size // 2
    disp_img = map_displacement_to_valid(H, W, p, dtype=image.dtype,
                                         device=image.device)
    dist_h, dist_w = disp_img[0].abs(), disp_img[1].abs()
    n_invalid = (dist_h + dist_w) * k_size - dist_h * dist_w
    r = (k_size ** 2) / (k_size ** 2 - n_invalid)

    # convolve
    z = F.conv2d(image, kernel, padding=p)

    # scale result by r
    z *= r[None, None, :, :]
    return z


def conv2d_extrap(image, kernel, returns_padded=False):
    """ extrapolation-based conv2d methods """
    # pad with zero
    k_size = kernel.shape[2]
    p = k_size // 2
    padded = F.pad(image, pad=(p, p, p, p))

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
    device = image.device
    padded[:, :, 0:p, :] = torch.tensor(interp1d(
        range(p, 2 * p + 1), padded[:, :, p:2 * p + 1, :].cpu().detach(),
        kind=mode, axis=2, fill_value="extrapolate")(range(0, p))).to(device)
    # bottom
    reverse = padded.flip(dims=(2,))
    padded[:, :, -p:, :] = torch.tensor(interp1d(
        range(p, 2 * p + 1), reverse[:, :, p:2 * p + 1, :].cpu().detach(),
        kind=mode, axis=2, fill_value="extrapolate")(range(p - 1, -1, -1))).to(
        device)
    # left
    padded[:, :, :, 0:p] = torch.tensor(interp1d(
        range(p, 2 * p + 1), padded[:, :, :, p:2 * p + 1].cpu().detach(),
        kind=mode, axis=3, fill_value="extrapolate")(range(0, p))).to(device)
    # right
    reverse = padded.flip(dims=(3,))
    padded[:, :, :, -p:] = torch.tensor(interp1d(
        range(p, 2 * p + 1), reverse[:, :, :, p:2 * p + 1].cpu().detach(),
        kind=mode, axis=3, fill_value="extrapolate")(range(p - 1, -1, -1))).to(
        device)

    if returns_padded:
        return F.conv2d(padded, kernel), padded
    return F.conv2d(padded, kernel)


def conv2d_rand(image, kernel, returns_padded=False):
    """ random methods """
    # pad with zero
    k_size = kernel.shape[2]
    p = k_size // 2
    padded = F.pad(image, pad=(p, p, p, p))

    N, C, H, W = image.shape
    # top
    value = image[:, :, 0:p + 1, :]
    mean = value.mean(dim=[2, 3])
    std = value.std(dim=[2, 3])
    pad = torch.randn((p, W + 2 * p), device=image.device)
    padded[:, :, 0:p, :] += \
        pad[None, None, :, :] * std[:, :, None, None] + mean[:, :, None, None]

    # bottom
    value = image[:, :, -(p + 1):, :]
    mean = value.mean(dim=[2, 3])
    std = value.std(dim=[2, 3])
    pad = torch.randn((p, W + 2 * p), device=image.device)
    padded[:, :, -p:, :] += \
        pad[None, None, :, :] * std[:, :, None, None] + mean[:, :, None, None]

    # left
    value = image[:, :, :, 0:p + 1]
    mean = value.mean(dim=[2, 3])
    std = value.std(dim=[2, 3])
    pad = torch.randn((H + 2 * p, p), device=image.device)
    padded[:, :, :, 0:p] += \
        pad[None, None, :, :] * std[:, :, None, None] + mean[:, :, None, None]

    # right
    value = image[:, :, :, -(p + 1):]
    mean = value.mean(dim=[2, 3])
    std = value.std(dim=[2, 3])
    pad = torch.randn((H + 2 * p, p), device=image.device)
    padded[:, :, :, -p:] += \
        pad[None, None, :, :] * std[:, :, None, None] + mean[:, :, None, None]

    # four corners added twice
    padded[:, :, 0:p, 0:p] /= 2
    padded[:, :, 0:p, -p:] /= 2
    padded[:, :, -p:, 0:p] /= 2
    padded[:, :, -p:, -p:] /= 2

    if returns_padded:
        return F.conv2d(padded, kernel), padded
    return F.conv2d(padded, kernel)


# DiffConv2d object
diff_conv2d_dict = {
    3: DiffConv2d(kernel_size=3),
    5: DiffConv2d(kernel_size=5),
    7: DiffConv2d(kernel_size=7)
}


def conv2d_diff(image, kernel):
    """ our diff-based cond2d method """
    k_size = kernel.shape[2]
    return diff_conv2d_dict[k_size].conv2d(image, kernel)


# methods to be considered
methods_dict = {
    'Zero': lambda image, kernel: conv2d_padding(
        image, kernel, padding_mode='constant'),
    'Refl': lambda image, kernel: conv2d_padding(
        image, kernel, padding_mode='reflect'),
    'Repl': lambda image, kernel: conv2d_padding(
        image, kernel, padding_mode='replicate'),
    'Circ': lambda image, kernel: conv2d_padding(
        image, kernel, padding_mode='circular'),
    'Extr': conv2d_extrap,
    'Rand': conv2d_rand,
    'Part': conv2d_part,
    'Diff': conv2d_diff
}

if __name__ == "__main__":
    x = torch.arange(0, 25.).reshape(1, 1, 5, 5)
    k = torch.ones((1, 1, 3, 3))
    _, pa = conv2d_rand(x, k, returns_padded=True)
    print(pa)
