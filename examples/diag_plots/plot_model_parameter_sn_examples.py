from pathlib import Path

import numpy as np
import aspect
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, rc_context
from matplotlib.patches import ConnectionPatch

import lime
from aspect.tools import detection_function, cosmic_ray_function
from lime import theme


# Configuration
fig_folder = Path('/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Emission line examples
n_pixels = 12           # pixels in each example box
sigma_pixels = 1      # x coordinate of the examples (velocity dispersion in pixels)
n_candidates = 50       # random realizations generated per intensity ratio
continuum_angle_range = (-45, 45)   # degrees, continuum tilt across the box
continuum_slope_scale = 1.0         # counts/pixel at a 45 degree tilt
seed = 42

# Selected realization index for each intensity ratio (rows from bottom to top)
example_idcs = {3:  [0, 1, 2, 3],
                5:  [0, 1, 6, 3],   #[0, 1, X, 3],
                10: [0, 1, 4, 16],   #[0, 1, 4, 16]
                15: [0, 1, 2, 3]}

preview = False         # True to plot all the candidates with their indices


def emission_candidates(int_ratio, n_samples, n_pixels, sigma, rng, angle_range=(-45, 45), slope_scale=1.0):
    """Gaussian lines on top of a tilted continuum and unit white noise.

    The continuum is a straight line pivoting on the box centre, with a random
    tilt (in degrees, within ``angle_range``) drawn independently for each
    sample. ``slope_scale`` sets the continuum slope, in counts/pixel, at a
    45 degree tilt.
    """
    x = np.arange(n_pixels)
    x0 = x - n_pixels // 2

    line = int_ratio * np.exp(-0.5 * np.square(x0 / sigma))

    angles = rng.uniform(angle_range[0], angle_range[1], n_samples)
    slopes = slope_scale * np.tan(np.radians(angles))
    continuum = slopes[:, None] * x0[None, :]

    return line[None, :] + continuum + rng.normal(0.0, 1.0, (n_samples, n_pixels))


def plot_candidates(data, int_ratio, ncols=10):
    """Grid with all the candidates and their index to select the examples."""
    nrows = int(np.ceil(data.shape[0] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.0))
    for i, ax_i in enumerate(axes.flat):
        ax_i.set_axis_off()
        if i < data.shape[0]:
            ax_i.step(np.arange(data.shape[1]), data[i], where='mid', color=aspect.cfg['colors']['emission'])
            ax_i.set_title(i, fontsize=8)
    fig.suptitle(rf'$A_{{line}}/\sigma_{{noise}}$ = {int_ratio}')
    plt.show()


# Data: parameter-space panel
x_detection = np.linspace(0, 10, 100)
y_detection = detection_function(x_detection)

x_pixel_lines = np.linspace(0, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

# Data: examples panel (independent seed per ratio so the indices do not change if a ratio is added)
int_ratios = sorted(example_idcs)
candidates = {ratio: emission_candidates(ratio, n_candidates, n_pixels, sigma_pixels, np.random.default_rng([seed, ratio]),
                                        continuum_angle_range, continuum_slope_scale)
              for ratio in int_ratios}

if preview:
    for ratio in int_ratios:
        plot_candidates(candidates[ratio], ratio)

# Style
theme.set_style('dark')
fig_cfg = theme.fig_defaults({'figure.dpi': 350, 'axes.labelsize': 16, 'axes.titlesize': 16})
colors = aspect.cfg['colors']

with rc_context(fig_cfg):

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.1)

    # ----- Left: parameter space -----
    ax = fig.add_subplot(gs[0])

    ax.plot(x_detection, y_detection, color=theme.colors['bg'])
    ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color=theme.colors['bg'])
    ax.fill_between(x_detection, y_detection, 10000, color=colors['emission'], edgecolor='none',
                    label='Line visibility region')
    ax.fill_between(x_detection, 0, y_detection, color=colors['continuum'], edgecolor='none',
                    label='Noise domination region')
    ax.scatter(np.full(len(int_ratios), sigma_pixels), int_ratios, color=theme.colors['fg'], s=20, zorder=5)

    ax.update({'xlabel': r'$\frac{\sigma_{line}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
               'ylabel': r'$\frac{A_{line}}{\sigma_{noise}}$ (Signal-to-noise proxy)'})
    ax.set_yscale('log')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.01, 10000)
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)

    # ----- Right: emission examples (one row per intensity ratio, bottom to top) -----
    n_rows, n_cols = len(int_ratios), max(len(idcs) for idcs in example_idcs.values())
    gs_ex = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1], hspace=0.15, wspace=0.15)

    for k, ratio in enumerate(int_ratios):
        row = n_rows - 1 - k
        # The two lower rows (weaker lines) are drawn in pink instead of the emission color
        line_color = colors['continuum'] if k < 2 else colors['emission']

        for col, idx in enumerate(example_idcs[ratio]):
            ax_ex = fig.add_subplot(gs_ex[row, col])
            ax_ex.step(np.arange(n_pixels), candidates[ratio][idx], where='mid', color=line_color, linewidth=2)
            ax_ex.set_xticks([])
            ax_ex.set_yticks([])

            # Connect the parameter space coordinate with the first example of the row
            if col == 0:
                con = ConnectionPatch(xyA=(sigma_pixels, ratio), coordsA='data', axesA=ax,
                                      xyB=(0, 0.5), coordsB='axes fraction', axesB=ax_ex,
                                      color=theme.colors['fg'], linestyle='--', linewidth=0.8)
                fig.add_artist(con)

        # Row label, to the right of the last example
        ax_ex.text(1.08, 0.5, rf'$A/\sigma={ratio}$', transform=ax_ex.transAxes,
                  ha='left', va='center', fontsize=10)

    plt.savefig(fig_folder/'diagnostic_plot_examples.png', bbox_inches='tight')