from pathlib import Path

import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import wget
from PIL import Image
from skimage.filters import gaussian, farid_h, farid_v
from skimage.util.shape import view_as_windows

from exp_utils import UNet, conv2d_methods

# set plot env
plt.style.use(['seaborn-paper'])
plt.rcParams.update({
    "xtick.major.pad": 2,
    "ytick.major.pad": 1,
    "font.family": "Times"
})

# set TeX
plt.rcParams["text.usetex"] = True
try:
    plt.text(0, 0, '$x$')
    plt.close()
except:
    # if latex is not installed, disable
    plt.rcParams["text.usetex"] = False

if __name__ == '__main__':
    # parameters
    filename = 'N45E045'
    h0 = 1700
    w0 = 800
    L = 192
    n_samples = 4
    sigma = 3.

    # download world map
    data_dir = Path(f'./exp_data/etopo/')
    fname = 'ETOPO_2022_v1_60s_N90W180_surface.tif'
    url = f'https://www.ngdc.noaa.gov/mgg/global/relief/' \
          f'ETOPO2022/data/60s/60s_surface_elev_gtif/{fname}'
    if not (data_dir / fname).exists():
        print(f'Download file: {fname}')
        wget.download(url, out=str(data_dir / fname), bar=None)
    else:
        print(f'Skip existing file: {fname}')
    Image.MAX_IMAGE_PIXELS = 10000000000
    tif_world = np.array(Image.open(data_dir / fname))

    # load tif
    dh = L * n_samples
    tif_clear = np.array(Image.open(
        f"./exp_data/etopo/ETOPO_2022_v1_15s_{filename}_surface.tif"))
    tif_blur = gaussian(tif_clear, sigma=sigma)
    image_clear = tif_clear[h0:h0 + dh, w0:w0 + dh]
    image_blur = tif_blur[h0:h0 + dh, w0:w0 + dh]

    # prediction
    patch_sizes = [192]
    methods = conv2d_methods.keys()
    image_clear_pred_all = {}
    for patch_size in patch_sizes:
        ps = patch_size
        patches_clear = view_as_windows(image_clear, [ps, ps], step=ps).reshape(
            -1, 1, ps, ps)
        patches_blur = view_as_windows(image_blur, [ps, ps], step=ps).reshape(
            -1, 1, ps, ps)
        patches_clear = torch.from_numpy(patches_clear).to(torch.float32)
        patches_blur = torch.from_numpy(patches_blur).to(torch.float32)
        p_min = patches_clear.min().item()
        p_max = patches_clear.max().item()
        diff = p_max - p_min
        patches_blur = 2 * (patches_blur - p_min) / diff - 1
        for method in methods:
            model = UNet(in_channels=1, out_channels=1,
                         conv2d_method=conv2d_methods[method],
                         seed=0, bias=True, activation=torch.nn.Tanh)
            res_dir = Path('./exp_results/etopo')
            model.load_state_dict(
                torch.load(res_dir / f'{method}_seed{0}.weights.pt',
                           map_location='cpu'))

            model.eval()
            with torch.no_grad():
                patches_clear_pred = model.forward(patches_blur)
            image_clear_pred = image_clear.copy()
            n = len(range(0, dh, patch_size))
            for i, i_loc in enumerate(range(0, dh, patch_size)):
                for j, j_loc in enumerate(range(0, dh, patch_size)):
                    image_clear_pred[i_loc:i_loc + ps, j_loc:j_loc + ps] = \
                        patches_clear_pred[i * n + j, 0]
            image_clear_pred_all[method + str(ps)] = \
                (image_clear_pred + 1) * diff / 2 + p_min

    # plot
    fig = plt.figure(dpi=300, figsize=(10, 6.7))
    outer = gridspec.GridSpec(2, 1, height_ratios=[.95, 1.6], hspace=.0)
    fontsize = 11.5

    #######
    # map #
    #######
    # cmap
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    colors_land = plt.cm.terrain(np.linspace(0.2, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = colors.LinearSegmentedColormap.from_list(
        'terrain_map', all_colors)
    divnorm_local = colors.TwoSlopeNorm(vmin=-1000., vcenter=0, vmax=4000)
    divnorm_world = colors.TwoSlopeNorm(vmin=-5000., vcenter=0, vmax=5000)

    # world
    y_title = -.2
    space_blur_clear = 25
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0],
        width_ratios=[dh * 2, dh, dh * 2 + space_blur_clear * n_samples])
    ax = plt.Subplot(fig, inner[0])
    ax.imshow(tif_world, norm=divnorm_world, cmap=terrain_map,
              extent=(-180, 180, 180, 0))
    ax.axis('off')
    ax.set_title(r"(a) Global topography at 15$''$ per pixel", y=y_title,
                 fontsize=fontsize)
    # local window rect
    lat_file = 90 - int(filename[1:3])
    lon_file = int(filename[4:7])
    pix_per_deg = 60 * 60 / 15
    deg_per_pix = 1 / pix_per_deg
    lat0 = lat_file + deg_per_pix * h0
    lon0 = lon_file + deg_per_pix * w0
    deg = deg_per_pix * dh
    ax.scatter(lon0 + deg / 2, lat0 + deg / 2, c='r', s=80, marker='+')
    fig.add_subplot(ax)

    # local
    ax = plt.Subplot(fig, inner[1])
    ax.imshow(image_clear, norm=divnorm_local, cmap=terrain_map)
    ax.axis('off')
    for crd in range(L, L * n_samples, L):
        ax.axvline(crd, lw=0.5, c='gray')
        ax.axhline(crd, lw=0.5, c='gray')
    hp = 1
    wp = 2
    rect = patches.Rectangle((L * wp, L * hp), L, L, linewidth=1.5,
                             edgecolor='r', facecolor='none', zorder=1000)
    ax.add_patch(rect)
    fig.add_subplot(ax)
    ax.set_title(r"(b) Shown area, %d$\times$%d patches" % (
        n_samples, n_samples), y=y_title, fontsize=fontsize)
    fig.add_subplot(ax)

    image_bc = np.full((L, L * 2 + space_blur_clear), np.nan)
    image_clear_train = image_clear[L * hp:L * (hp + 1), L * wp:L * (wp + 1)]
    image_blur_train = image_blur[L * hp:L * (hp + 1), L * wp:L * (wp + 1)]
    image_bc[:, :L] = image_blur_train
    image_bc[:, -L:] = image_clear_train
    ax = plt.Subplot(fig, inner[2])
    ax.imshow(image_bc, norm=divnorm_local, cmap=terrain_map)
    ax.axis('off')
    tl = ax.text(L - 5, 5, 'Low-res input', fontsize=fontsize,
                 ha='right', va='top')
    th = ax.text(L + space_blur_clear + L - 5, 5, 'High-res output',
                 fontsize=fontsize, ha='right', va='top')
    tl.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))
    th.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))
    ax.set_title(r"(c) A patch for training, %d$\times$%d pixels" % (L, L),
                 y=y_title, fontsize=fontsize)
    fig.add_subplot(ax)

    # #########
    # # error #
    # #########
    inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[1],
                                             hspace=.1)

    ps = 192
    caption = 'd'
    yt = -0.2
    for i_method, method in enumerate(list(methods)[:-1]):
        image_clear_pred = image_clear_pred_all[method + str(ps)]
        if i_method == 0:
            ax = plt.Subplot(fig, inner[0, 0])
            ax.imshow(image_clear_pred, norm=divnorm_local, cmap=terrain_map)
            ax.axis('off')
            ax.set_title(f'({caption}) Prediction by $\\texttt{{{method}}}$',
                         fontsize=fontsize, y=yt)
            caption = chr(ord(caption) + 1)
            fig.add_subplot(ax)
        error = image_clear - image_clear_pred
        error_v = farid_v(farid_v(error))
        error_h = farid_h(farid_h(error))
        error_vh = (error_v + error_h) / 2
        ax = plt.Subplot(fig, inner[(i_method + 1) // 5, (i_method + 1) % 5])
        vm = 1
        ax.imshow(error_vh, cmap='bwr', vmin=-vm, vmax=vm)
        ax.axis('off')
        ax.set_title(f'({caption}) Error by $\\texttt{{{method}}}$',
                     fontsize=fontsize, y=yt)
        caption = chr(ord(caption) + 1)
        fig.add_subplot(ax)

    # save
    out_dir = Path('./exp_results/etopo')
    plt.savefig(out_dir / 'etopo.pdf', pad_inches=0.1, bbox_inches='tight',
                facecolor='w')
