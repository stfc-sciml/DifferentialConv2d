import argparse
from pathlib import Path

import numpy as np
import torch
import wget
from PIL import Image
from mpi4py import MPI
from skimage.filters import gaussian
from skimage.util.shape import view_as_windows
from tqdm import tqdm

from exp_utils import UNet, conv2d_methods


def getXY_from_a_file(lat_lon_str, gaussian_sigma, tiff_truncate, patch_size,
                      returns_min_max=False):
    """ get patches from a file """
    # read
    img_clear = np.array(Image.open(
        f"./exp_data/etopo/ETOPO_2022_v1_15s_{lat_lon_str}_surface.tif"))
    # blur
    img_blur = gaussian(img_clear, sigma=gaussian_sigma)

    # truncate edges to avoid any edge effects from gaussian filter
    tt = tiff_truncate
    img_clear = img_clear[tt:-tt, tt:-tt]
    img_blur = img_blur[tt:-tt, tt:-tt]

    # image to patches
    ps = patch_size
    patches_clear = view_as_windows(img_clear, [ps, ps],
                                    step=ps).reshape(-1, 1, ps, ps)
    patches_blur = view_as_windows(img_blur, [ps, ps],
                                   step=ps).reshape(-1, 1, ps, ps)

    # to tensor
    patches_clear = torch.from_numpy(patches_clear).to(torch.float32)
    patches_blur = torch.from_numpy(patches_blur).to(torch.float32)

    # per patch normalization
    n = len(patches_clear)
    p_min, _ = patches_clear.view(n, -1).min(dim=1)
    p_max, _ = patches_clear.view(n, -1).max(dim=1)
    diff = p_max - p_min
    diff = torch.clamp(diff, min=1e-10)  # so a uniform patch is filled by -1
    patches_clear = \
        2 * (patches_clear - p_min[:, None, None, None]
             ) / diff[:, None, None, None] - 1
    patches_blur = \
        2 * (patches_blur - p_min[:, None, None, None]
             ) / diff[:, None, None, None] - 1

    # return patches and ranges
    if returns_min_max:
        return patches_blur, patches_clear, p_min, p_max
    else:
        return patches_blur, patches_clear


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('-m', '--methods', type=str, nargs='+',
                        choices=conv2d_methods.keys(), required=True,
                        help='methods for Conv2d')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='device')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='epochs')
    parser.add_argument('-l', '--lr', type=float, default=0.00001,
                        help='learning rate')
    parser.add_argument('-E', '--step-size-lr', type=int, default=10,
                        help='epoch to reduce lr')
    parser.add_argument('-G', '--gamma-lr', type=float, default=0.1,
                        help='rate to reduce lr')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-f', '--n-files-per-group', type=int, default=24,
                        help='number of lat-lon files grouped together'
                             'for sampling batches')
    parser.add_argument('-T', '--tiff-truncate', type=int, default=264,
                        help='truncate tiff to avoid edge effects '
                             'from Gaussian blur')
    parser.add_argument('-L', '--patch-size', type=int, default=192,
                        help='patch size')
    parser.add_argument('-S', '--sigma', type=float, default=3.,
                        help='sigma of Gaussian filter')
    parser.add_argument('-W', '--frame-width-eval', type=int, default=8,
                        help='frame width for evaluation')
    parser.add_argument('-P', '--patch-sizes-eval', type=int, nargs='+',
                        default=[64, 128, 192, 256],
                        help='patch sizes for evaluation')
    args = parser.parse_args()

    # mpi
    mpi_size = MPI.COMM_WORLD.Get_size()
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    assert mpi_size == len(args.methods)

    # download dataset
    data_dir = Path(f'./exp_data/etopo/')
    data_dir.mkdir(exist_ok=True, parents=True)
    lats = ['N90', 'N75', 'N60', 'N45', 'N30', 'N15', 'N00',
            'S15', 'S30', 'S45', 'S60', 'S75']
    lons = ['W180', 'W165', 'W150', 'W135', 'W120', 'W105', 'W090',
            'W075', 'W060', 'W045', 'W030', 'W015',
            'E000', 'E015', 'E030', 'E045', 'E060', 'E075', 'E090',
            'E105', 'E120', 'E135', 'E150', 'E165']
    lat_lon_list = []
    for lat in lats:
        for lon in lons:
            lat_lon_list.append(f'{lat}{lon}')
            if mpi_rank == 0:
                fname = f'ETOPO_2022_v1_15s_{lat}{lon}_surface.tif'
                url = f'https://www.ngdc.noaa.gov/mgg/global/relief/' \
                      f'ETOPO2022/data/15s/15s_surface_elev_gtif/{fname}'
                if not (data_dir / fname).exists():
                    print(f'Download file: {fname}')
                    wget.download(url, out=str(data_dir / fname), bar=None)
                else:
                    print(f'Skip existing file: {fname}')
    MPI.COMM_WORLD.Barrier()
    lat_lon_list = np.array(lat_lon_list)

    # model
    method = args.methods[mpi_rank]
    device = args.device if mpi_size == 1 else f'cuda:{mpi_rank}'
    model = UNet(in_channels=1, out_channels=1,
                 conv2d_method=conv2d_methods[method],
                 seed=args.seed, bias=True,
                 activation=torch.nn.Tanh).to(device)

    # optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size_lr, gamma=args.gamma_lr)
    loss_func = torch.nn.MSELoss()

    # results
    res_dir = Path('./exp_results/etopo')
    res_dir.mkdir(exist_ok=True)
    step_losses = []
    epoch_losses = []

    # sizes
    nF = len(lat_lon_list)  # number of files
    nFpG = args.n_files_per_group  # number of files per group
    assert nF % nFpG == 0
    nG = nF // nFpG  # number of groups

    L = args.patch_size
    T = args.tiff_truncate
    assert (3600 - 2 * T) % L == 0
    nPpF = ((3600 - 2 * T) // L) ** 2  # number of patches per file
    nPpB = args.batch_size  # number of patches per batches
    assert nPpF % nPpB == 0
    nBpF = nPpF // nPpB  # number of batches per file

    # training
    model.train()
    disable_pbar = mpi_size > 1
    for epoch in range(args.epochs):
        if disable_pbar:
            print(f'Rank {mpi_rank}: {method}, epoch {epoch + 1}')
        # permute order of lat_lon files
        np.random.seed(epoch)
        perm = np.random.permutation(nF)
        lat_lon_perm = lat_lon_list[perm]
        epoch_loss = 0.

        # loop over file groups
        pbar_group = tqdm(range(nG), desc=f'EPOCH {epoch + 1}',
                          disable=disable_pbar)
        for i_group in pbar_group:
            X_group, Y_group = [], []
            # loop over files
            for i_ll, ll in enumerate(
                    lat_lon_perm[i_group * nFpG:(i_group + 1) * nFpG]):
                X_file, Y_file = getXY_from_a_file(
                    ll, gaussian_sigma=args.sigma,
                    tiff_truncate=T, patch_size=L)
                X_group.append(X_file)
                Y_group.append(Y_file)
                pbar_group.set_postfix_str(
                    f'loading file {i_ll + 1} / {nFpG}')
            X_group = torch.cat(X_group, dim=0)
            Y_group = torch.cat(Y_group, dim=0)
            # permute patches in group
            np.random.seed(epoch * nG + i_group)
            assert len(X_group) == nPpF * nFpG
            perm = np.random.permutation(len(X_group))
            X_group = X_group[perm]
            Y_group = Y_group[perm]

            # loop over batches
            for i_batch in range(nBpF * nFpG):
                X = X_group[i_batch * nPpB:(i_batch + 1) * nPpB]
                Y = Y_group[i_batch * nPpB:(i_batch + 1) * nPpB]
                X, Y = X.to(device), Y.to(device)
                # loss
                Y_pred = model.forward(X)
                loss = loss_func(Y_pred, Y)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # log
                epoch_loss += loss.item()
                step_losses.append(loss.item())
                pbar_group.set_postfix_str(
                    f'training batch {i_batch + 1} / {nBpF * nFpG}, '
                    f'loss={loss.item():.4f}')
        epoch_loss /= nF * nBpF
        epoch_losses.append(epoch_loss)
        scheduler.step()

    # save model
    torch.save(
        model.state_dict(),
        res_dir / f'{method}_seed{args.seed}.weights.pt')

    # evaluate
    model.eval()
    whole_sum = {L: 0. for L in args.patch_sizes_eval}
    inter_sum = {L: 0. for L in args.patch_sizes_eval}
    frame_sum = {L: 0. for L in args.patch_sizes_eval}
    d = args.frame_width_eval

    # loop over patch sizes
    for L in args.patch_sizes_eval:
        assert (3600 - 2 * T) % L == 0
        nPpF = ((3600 - 2 * T) // L) ** 2  # number of patches per file
        nPpB = nPpF // 8  # number of patches per batches
        assert nPpF % nPpB == 0
        nBpF = nPpF // nPpB  # number of batches per file
        nP = nPpF * nF
        pbar_file = tqdm(lat_lon_list, desc=f'EVAL, patch size {L}',
                         disable=disable_pbar)
        if disable_pbar:
            print(f'Rank {mpi_rank}: {method}, EVAL, patch size {L}')
        for ll in pbar_file:
            X_file, Y_file = getXY_from_a_file(
                ll, gaussian_sigma=args.sigma,
                tiff_truncate=T, patch_size=L)
            for i_batch in range(nBpF):
                X = X_file[i_batch * nPpB:(i_batch + 1) * nPpB]
                Y = Y_file[i_batch * nPpB:(i_batch + 1) * nPpB]
                X, Y = X.to(device), Y.to(device)
                with torch.no_grad():
                    Y_pred = model.forward(X)
                whole = ((Y - Y_pred) ** 2).sum().item()
                inter = ((Y[:, :, d:-d, d:-d] -
                          Y_pred[:, :, d:-d, d:-d]) ** 2).sum().item()
                frame = whole - inter
                whole_sum[L] += whole
                inter_sum[L] += inter
                frame_sum[L] += frame
        whole_sum[L] /= nP * L ** 2
        inter_sum[L] /= nP * (L - 2 * d) ** 2
        frame_sum[L] /= nP * (L ** 2 - (L - 2 * d) ** 2)

    # save history
    torch.save(
        {'whole_loss': whole_sum,
         'inter_loss': inter_sum,
         'frame_loss': frame_sum,
         'step_losses': step_losses,
         'epoch_losses': epoch_losses},
        res_dir / f'{method}_seed{args.seed}.hist.pt',
    )
