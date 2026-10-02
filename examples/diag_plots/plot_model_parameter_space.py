from pathlib import Path

import numpy as np
import aspect
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt, rc_context
from aspect.tools import detection_function, cosmic_ray_function
from lime import theme


# Configuration
fig_folder = Path('/home/vital/Dropbox/Astrophysics/Tools/aspect')

# -- parameter-space panel --
cfg_file = '../../training/12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
params = aspect.load_cfg(cfg_file)['properties_12_pixels_v12']
int_ratio_max = params['int_ratio_max']

# -- categories panel --
version = '12_pixels_v8'
output_folder = Path('/home/vital/Astrodata/aspect/medium_box/PorSi/')

# Data: parameter-space panel
x_detection = np.linspace(0, 10, 100)
y_detection = detection_function(x_detection)

x_pixel_lines = np.linspace(0, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

continuum_limit = params['white_noise']['max_int_ratio']

int_cosmic_ray = np.linspace(params['doublet_em']['min_int_ratio'], int_ratio_max, 100)
res_cosmic_ray = cosmic_ray_function(int_cosmic_ray, res_ratio_check=False)

x_double_min, x_doublet_max = params['doublet_em']['min_res_ratio'], params['doublet_em']['max_res_ratio']
y_doublet_min, y_doublet_max = params['doublet_em']['min_int_ratio'], params['doublet_em']['max_int_ratio']
res_doublet = np.linspace(x_double_min, x_doublet_max, 100)

# ---------------------------------------------------------------------------
# Data: categories panel
# ---------------------------------------------------------------------------
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
data_matrix = data_matrix[:, -12:]
x_range = np.arange(12)

target_idcs = {'white-noise': 116, 'continuum': 32,
               'cosmic-ray': 1, 'dead-pixel': 34,
               'emission': 81, 'absorption': 109,
               'doublet-em': 46, 'doublet-abs': 129}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# theme.set_style('dark')
fig_cfg = theme.fig_defaults({'figure.dpi': 350, 'axes.labelsize': 16,
                               'axes.titlesize': 16, 'hatch.linewidth': 0.3,
                               'legend.fontsize': 8})

with rc_context(fig_cfg):

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1], wspace=0.1)

    # ----- Left: parameter space -----
    ax = fig.add_subplot(gs[0])

    ax.plot(x_detection, y_detection, color=theme.colors['fg'], label='Detection boundary')
    ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color=theme.colors['fg'],
            label='Single pixel boundary')

    ax.fill_between(x_detection, y_detection, 10000, color=aspect.cfg['colors']['emission'], label='emission', edgecolor='none')
    ax.fill_between(x_detection, 0, continuum_limit, color=aspect.cfg['colors']['white-noise'], label='white-noise', edgecolor='none')
    ax.fill_between(x_detection, continuum_limit, y_detection, color=aspect.cfg['colors']['continuum'], label='continuum', edgecolor='none')

    ax.fill_betweenx(int_cosmic_ray, 0, res_cosmic_ray, color=aspect.cfg['colors']['cosmic-ray'], label='cosmic-ray', edgecolor='none')

    ax.fill_between(res_doublet, y_doublet_min, y_doublet_max, color=aspect.cfg['colors']['doublet-em'], alpha=0.8, label='doublet')

    ax.update({'xlabel': r'$\frac{\sigma_{line}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
               'ylabel': r'$\frac{A_{line}}{\sigma_{noise}}$ (Signal-to-noise proxy)'})
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.08), ncol=3, framealpha=0.95, fontsize=10)

    ax.set_yscale('log')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.01, 10000)

    # Upper axis
    ax2 = ax.twiny()
    ticks_values = ax.get_xticks()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values * 6]
    ax2.set_xticks(ticks_values)
    ax2.set_xticklabels(ticks_labels)
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')

    # Grid
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)

    # ----- Right: categories (nrows=4, ncols=2) -----
    gs_cat = gridspec.GridSpecFromSubplotSpec(4, 2, subplot_spec=gs[1], hspace=0.6, wspace=0.15)

    axes = []
    for i, (comp, idx) in enumerate(target_idcs.items()):
        row, col = divmod(i, 2)
        axes.append(fig.add_subplot(gs_cat[row, col]))

        y_arr = data_matrix[idx, :]
        axes[i].step(x_range, y_arr, where='mid', color=aspect.cfg['colors'][comp], linewidth=2)
        axes[i].set_title(comp.capitalize(), fontsize=11)

        axes[i].set_ylabel('')
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])
        axes[i].set_xticklabels([])

    plt.savefig(fig_folder/'diagnostic_plot_new.png', bbox_inches='tight')
    # plt.tight_layout()
    # plt.savefig('./diagnostic_plot_combined.png')
    # plt.show()

# from pathlib import Path
#
# import numpy as np
# import aspect
# from matplotlib import pyplot as plt, rc_context
# from aspect.tools import detection_function, cosmic_ray_function
# from lime import theme
#
#
# # Configuration file
# cfg_file = '../../training/12_pixels.toml'
# sample_cfg = aspect.load_cfg(cfg_file)
# params = aspect.load_cfg(cfg_file)['properties_12_pixels_v12']
# fig_folder = Path('/home/vital/Dropbox/Astrophysics/Tools/aspect')
#
# # Parameters
# int_ratio_max = params['int_ratio_max']
#
# # Compute the plot values
# x_detection = np.linspace(0, 10, 100)
# y_detection = detection_function(x_detection)
#
# x_pixel_lines = np.linspace(0, 0.6, 100)
# y_pixel_lines = cosmic_ray_function(x_pixel_lines)
# idcs_crop = y_pixel_lines > 5
#
# # theme.set_style('dark')
# fig_cfg = theme.fig_defaults({'figure.dpi': 350, 'axes.labelsize': 16,
#                               'axes.titlesize': 16, 'figure.figsize': (6, 6),
#                               'hatch.linewidth': 0.3, "legend.fontsize" : 8})
#
# # Continuum
# continuum_limit = params['white_noise']['max_int_ratio']
#
# # Cosmic rays
# int_cosmic_ray = np.linspace(params['doublet_em']['min_int_ratio'], int_ratio_max, 100)
# res_cosmic_ray = cosmic_ray_function(int_cosmic_ray, res_ratio_check=False)
#
# # Doublet
# x_double_min, x_doublet_max = params['doublet_em']['min_res_ratio'], params['doublet_em']['max_res_ratio']
# y_doublet_min, y_doublet_max = params['doublet_em']['min_int_ratio'], params['doublet_em']['max_int_ratio']
# res_doublet = np.linspace(x_double_min, x_doublet_max, 100)
#
# # Plot
# # theme.set_style('dark')
# with rc_context(fig_cfg):
#
#     fig, ax = plt.subplots()
#
#     # Detection boundary
#     ax.plot(x_detection, y_detection, color=theme.colors['fg'], label='Detection boundary')
#
#     # Single pixel line boundary
#     ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color=theme.colors['fg'],
#             label='Single pixel boundary')
#
#     # Positive and negative detection
#     ax.fill_between(x_detection, y_detection, 10000, color=aspect.cfg['colors']['emission'], label='emission', edgecolor='none')
#     ax.fill_between(x_detection, 0, continuum_limit, color=aspect.cfg['colors']['white-noise'], label='white-noise', edgecolor='none')
#     ax.fill_between(x_detection, continuum_limit, y_detection, color=aspect.cfg['colors']['continuum'], label='continuum', edgecolor='none')
#
#     # Cosmic and single-pixel lines
#     ax.fill_betweenx(int_cosmic_ray, 0, res_cosmic_ray, color=aspect.cfg['colors']['cosmic-ray'], label='cosmic-ray', edgecolor='none')
#
#     # Doublet
#     ax.fill_between(res_doublet, y_doublet_min, y_doublet_max, color=aspect.cfg['colors']['doublet-em'], alpha=0.8, label='doublet')
#
#     # Wording
#     ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
#                'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
#     ax.legend(loc='lower center', ncol=3, framealpha=0.95, fontsize=10)
#
#     # Axis format
#     ax.set_yscale('log')
#     ax.set_xlim(0, 3)
#     ax.set_ylim(0.01, 10000)
#
#     # Upper axis
#     ax2 = ax.twiny()
#     ticks_values = ax.get_xticks()
#     ticks_labels = [f'{tick:.0f}' for tick in ticks_values*6]
#     ax2.set_xticks(ticks_values)
#     ax2.set_xticklabels(ticks_labels)
#     ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')
#
#     # Grid
#     ax.grid(axis='x', color='0.95', zorder=1)
#     ax.grid(axis='y', color='0.95', zorder=1)
#
#     plt.tight_layout()
#     plt.savefig(fig_folder/'diagnostic_plot.png')
#     plt.show()
