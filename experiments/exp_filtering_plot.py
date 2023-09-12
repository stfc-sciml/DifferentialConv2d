from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import chebyu, sph_harm

from exp_filtering_methods import methods_dict, conv2d_padding

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


def latex_float(f):
    """ 1e5 => 10^5 """
    float_str = "{0:.1e}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str


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


def f_chebyu_fix(n, gap):
    """ Chebyshev on fixed domain """
    delta = 0.001
    h_crd = np.arange(.3 - delta * gap, .8 + delta * gap + 1e-9, delta)
    w_crd = np.arange(.3 - delta * gap, .8 + delta * gap + 1e-9, delta)
    h, w = np.meshgrid(h_crd, w_crd, indexing='ij')
    un = chebyu(n)
    return un(h) * un(w) * np.sin((h + w) * n)


def unf(n, x):
    """ Chebyshev wrapper """
    return chebyu(n)(x)


def Laplace_f_chebyu_fix(n, gap):
    """ Laplacian of Chebyshev on fixed domain """
    delta = 0.001
    h_crd = np.arange(.3 - delta * gap, .8 + delta * gap + 1e-9, delta)
    w_crd = np.arange(.3 - delta * gap, .8 + delta * gap + 1e-9, delta)
    h, w = np.meshgrid(h_crd, w_crd, indexing='ij')

    # analytical solution from mathematica
    m_sol = '-((2*n*((1+n)*unf[-1+n,h]-h*n*unf[n,h])' \
            '*unf[n,w]*Cos[n*(h+w)])/(-1+h^2))-(2*n*unf[n,h]' \
            '*((1+n)*unf[-1+n,w]-n*w*unf[n,w])*Cos[n*(h+w)])/' \
            '(-1+w^2)-2*n^2*unf[n,h]*unf[n,w]*Sin[n*(h+w)]+' \
            '(unf[n,w]*((2+n)*(-n+h^2*(3+n))*unf[n,h]-3*h*(1+n)' \
            '*unf[1+n,h])*Sin[n*(h+w)])/(-1+h^2)^2+(unf[n,h]' \
            '*((2+n)*(-n+(3+n)*w^2)*unf[n,w]-3*(1+n)*w*unf' \
            '[1+n,w])*Sin[n*(h+w)])/(-1+w^2)^2'
    m_sol = m_sol.replace('[', '(').replace(']', ')').replace('^', '**')
    m_sol = m_sol.replace('Sin', 'np.sin').replace('Cos', 'np.cos')
    return eval(m_sol)


if __name__ == '__main__':
    fig = plt.figure(dpi=300, figsize=(10, 9))
    top = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=.22)
    outer = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=top[0],
                                             height_ratios=[1.5, .07, 1],
                                             hspace=.15, wspace=0.24)
    fontsize = 13

    ##########
    # images #
    ##########
    y_title = -.3
    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 0],
                                             wspace=0.1, hspace=0.1)
    for i_order, order in enumerate([1, 10, 50, 100]):
        ax = plt.Subplot(fig, inner[i_order // 2, i_order % 2])
        ax.imshow(f_chebyu(order), origin='lower', cmap='turbo')
        ax.axis('off')
        t = ax.text(10, 10, '$C_{%d}$' % order, fontsize=fontsize,
                    ha='left', va='bottom', c='k')
        t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))
        fig.add_subplot(ax)
    ax = plt.Subplot(fig, outer[1, 0])
    ax.set_title('(a) Chebyshev $C_n$', y=y_title, fontsize=fontsize)
    ax.axis('off')
    fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 1],
                                             wspace=0.1, hspace=0.1)
    for i_order, order in enumerate([5, 10, 20, 50]):
        ax = plt.Subplot(fig, inner[i_order // 2, i_order % 2])
        ax.imshow(f_sph(order, order * 2), origin='lower', cmap='turbo')
        ax.axis('off')
        t = ax.text(10, 10, '$S_{%d}$' % order,
                    fontsize=fontsize,
                    ha='left', va='bottom', c='k')
        t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))
        fig.add_subplot(ax)
    ax = plt.Subplot(fig, outer[1, 1])
    ax.set_title('(b) Spherical harmonics $S_{n}$', y=y_title,
                 fontsize=fontsize)
    ax.axis('off')
    fig.add_subplot(ax)

    inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, 2],
                                             wspace=0.1, hspace=0.1)
    bh = [0, 33, 66, 99]
    u = f_NS(batches=bh, times=0)
    for i_order, order in enumerate(bh):
        ax = plt.Subplot(fig, inner[i_order // 2, i_order % 2])
        ax.imshow(u[i_order], origin='lower', cmap='turbo')
        ax.axis('off')
        fig.add_subplot(ax)
    ax = plt.Subplot(fig, outer[1, 2])
    ax.set_title('(c) Navier–Stokes snapshots', y=y_title, fontsize=fontsize)
    ax.axis('off')
    fig.add_subplot(ax)

    #########
    # error #
    #########
    for i_data, (data_name, caption, title, crd_name) in enumerate(
            zip(['chebyu', 'sph', 'NS'],
                ['d', 'e', 'f'],
                ['Chebyshev', 'spherical harmonics', 'Navier–Stokes'],
                ['Function order $n$ in $C_n$',
                 'Function order $n$ in $S_{n}$',
                 'Batch \# in dataset'])):
        res = torch.load(f'./exp_results/filtering/{data_name}.pt')
        ax = plt.Subplot(fig, outer[2, i_data])
        for i_method, (name, method) in enumerate(methods_dict.items()):
            ax.plot(res['crd'], res['err'][:, i_method],
                    label=f'$\\texttt{{{name}}}$',
                    lw=1.5 if name == 'Diff' else 1.0,
                    c='k' if name == 'Diff' else None)
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_xticks(ticks=res['crd'], labels=[
            str(res['crd'][i]) if i % 2 == 0 else ''
            for i in range(len(res['crd']))])
        ax.set_title('(%s) $\epsilon^1$ for {%s}' % (
            caption, title), y=-.5, fontsize=fontsize)
        ax.set_xlim(res['crd'][0], res['crd'][-1])
        ax.set_xlabel(crd_name, fontsize=fontsize)
        ax.minorticks_off()
        if i_data == 0:
            ax.set_ylabel('$L^1$ error, $\epsilon^1$', fontsize=fontsize)
            ax.set_ylim(1e-8, 10)
            ax.legend(fontsize=fontsize, ncol=2, handlelength=1.,
                      labelspacing=0.1, columnspacing=.8, loc=(.26, 0.02),
                      borderpad=.2)
        if i_data == 2:
            ax.set_ylim(1e-2, 10)
        if i_data == 1:
            ax.set_ylim(1e-7, 1)
        fig.add_subplot(ax)

    #############
    # artefacts #
    #############
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=top[1])

    # color map
    cmap = matplotlib.colormaps['Spectral_r'].copy()
    cmap.set_over('magenta')
    cmap.set_under('blue')
    vrange = .2

    # analytical truth
    d = 0.001
    order = 100
    analytical = Laplace_f_chebyu_fix(order, 0) * d ** 2

    # setup zoom window
    keys = (['Truth', 'Diff', 'Extr', 'Repl', 'Zero'][::-1])
    n_zoom = len(keys)
    dh_zoom = 10
    space_zoom = 2
    h_zoom = n_zoom * dh_zoom + (n_zoom - 1) * space_zoom
    w_zoom = int(h_zoom / analytical.shape[0] * analytical.shape[1])

    ax = plt.Subplot(fig, inner[0])
    ax.imshow(analytical, origin='lower', cmap=cmap, vmin=-vrange, vmax=vrange,
              extent=[.3, .8, .3, .8])
    # annotation
    rect = patches.Rectangle((.8 - d * w_zoom, .8 - d * dh_zoom),
                             (w_zoom - 1) * d, (dh_zoom - 1) * d,
                             linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)
    ax.annotate("Zoom",
                xy=(.8 - d * w_zoom - 0.01, .8 - d * dh_zoom - 0.01),
                xytext=(.8 - d * w_zoom - .2, .8 - d * dh_zoom - .15),
                arrowprops=dict(arrowstyle="-|>", color='k', lw=1), c='k',
                fontsize=fontsize)
    ax.axis('off')
    ax.set_title('(%s) Analytical $\Delta C_{%d}$' % ('g', order), y=-.2,
                 fontsize=fontsize)
    fig.add_subplot(ax)
    # color bar
    cm = plt.cm.ScalarMappable(cmap=cmap)
    cm.set_clim(-vrange, vrange)
    cax = fig.add_axes([0.065, 0.14, 0.012, 0.17])
    cbar = fig.colorbar(cm, ticks=[-vrange, -vrange / 2, 0,
                                   vrange / 2, vrange],
                        orientation='vertical', cax=cax, extend='both',
                        extendfrac=.1)
    cbar.ax.tick_params(labelsize=fontsize)

    # laplace kernels
    laplace_kernels = [
        torch.tensor([[0, 1., 0], [1, -4, 1], [0, 1, 0]]),
        torch.tensor([[0, 0, -(1 / 12), 0, 0],
                      [0, 0, 4 / 3, 0, 0],
                      [-(1 / 12), 4 / 3, -5, 4 / 3, -(1 / 12)],
                      [0, 0, 4 / 3, 0, 0],
                      [0, 0, -(1 / 12), 0, 0]]),
        torch.tensor([[0, 0, 0, 1 / 90, 0, 0, 0],
                      [0, 0, 0, -(3 / 20), 0, 0, 0],
                      [0, 0, 0, 3 / 2, 0, 0, 0],
                      [1 / 90, -(3 / 20), 3 / 2, -(49 / 9), 3 / 2, -(3 / 20),
                       1 / 90],
                      [0, 0, 0, 3 / 2, 0, 0, 0],
                      [0, 0, 0, -(3 / 20), 0, 0, 0],
                      [0, 0, 0, 1 / 90, 0, 0, 0]])
    ]

    # loop over kernel size
    for i_k, (lap_kernel, caption) in enumerate(
            zip(laplace_kernels, ['h', 'i', 'j'])):
        # ground truth
        lap_kernel = lap_kernel.unsqueeze(0).unsqueeze(0)
        ksize = lap_kernel.shape[2]
        p = ksize // 2
        image = torch.tensor(f_chebyu_fix(order, p)).to(torch.float).unsqueeze(
            0).unsqueeze(0)
        z_true = conv2d_padding(image, lap_kernel, ignore_padding=True)

        # crop image by p pixels
        image_crop = image[:, :, p:-p, p:-p]
        _, _, hc, wc = image_crop.shape
        n_inv = hc * wc - (hc - p * 2) * (wc - p * 2)

        # loop over methods
        solutions = {'Truth': z_true}
        errors = {}
        for i_method, (name, method) in enumerate(methods_dict.items()):
            z_pred = method(image_crop, lap_kernel)
            solutions[name] = z_pred
            errors[name] = (z_pred - z_true).abs().sum() / n_inv

        # merge zooms
        pic = torch.full((h_zoom, w_zoom), float('nan'))
        offset = dh_zoom + space_zoom
        for i, key in enumerate(keys):
            pic[offset * i:offset * i + dh_zoom, :] = \
                solutions[key][0, 0, -dh_zoom:, -w_zoom:]

        # plot
        ax = plt.Subplot(fig, inner[i_k + 1])
        ax.imshow(pic, origin='lower', cmap=cmap, vmin=-vrange, vmax=vrange)
        for i, key in enumerate(keys):
            if key != 'Truth':
                err = errors[key]
                ax.text(.5, offset * i + 1, f'$\\texttt{{{key}}}$',
                        fontsize=fontsize)
                ax.text(w_zoom * .9, offset * i + 1,
                        f'$\epsilon^1$=' + latex_float(err), fontsize=fontsize,
                        ha='right')
            else:
                ax.text(.5, offset * i + 1, f'{key}', fontsize=fontsize)
        ax.axis('off')
        ax.set_title(
            '(%s) $\Delta C_{%d}$ by conv, $K=%d$' % (caption, order, ksize),
            y=-.2, fontsize=fontsize)
        fig.add_subplot(ax)

    # save
    out_dir = Path('./exp_results/filtering')
    plt.savefig(out_dir / 'filter.pdf', pad_inches=0.1, bbox_inches='tight',
                facecolor='w')
