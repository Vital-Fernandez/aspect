from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, rc_context
import aspect
from aspect.tools import detection_function, cosmic_ray_function, broad_component_function
from lime import theme


# Output plot
fig_folder = Path('/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Configuration file
cfg_file = '../../training/12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
params = aspect.load_cfg(cfg_file)['properties_12_pixels_v7']

# Parameters
int_ratio_max = params['int_ratio_max']

# Compute the plot values
x_detection = np.linspace(0, 10, 100)
y_detection = detection_function(x_detection)

x_pixel_lines = np.linspace(0, 0.6, 100)
y_pixel_lines = cosmic_ray_function(x_pixel_lines)
idcs_crop = y_pixel_lines > 5

# theme.set_style('dark')
fig_cfg = theme.fig_defaults({'figure.dpi': 350,
                              'axes.labelsize': 16,
                              'axes.titlesize': 16,
                              'figure.figsize': (6, 6),
                              'hatch.linewidth': 0.3,
                              "legend.fontsize" : 8})

# Continuum
continuum_limit = params['white_noise']['max_int_ratio']

# Cosmic rays
int_cosmic_ray = np.linspace(params['doublet']['min_int_ratio'], int_ratio_max, 100)
res_cosmic_ray = cosmic_ray_function(int_cosmic_ray, res_ratio_check=False)

# Doublet
x_double_min, x_doublet_max = params['doublet']['min_res_ratio'], params['doublet']['max_res_ratio']  # x-axis boundaries
y_doublet_min, y_doublet_max = params['doublet']['min_int_ratio'], params['doublet']['max_int_ratio']  # y-axis boundaries
res_doublet = np.linspace(x_double_min, x_doublet_max, 100)

# Plot
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    # Detection boundary
    ax.plot(x_detection, y_detection, color='black', label='Detection boundary')

    # Single pixel line boundary
    ax.plot(x_pixel_lines[idcs_crop], y_pixel_lines[idcs_crop], linestyle='--', color='black',
            label='Single pixel boundary')

    # Positive and negative detection
    ax.fill_between(x_detection, y_detection, 10000, color=aspect.cfg['colors']['emission'], label='emission', edgecolor='none')
    ax.fill_between(x_detection, 0, continuum_limit, color=aspect.cfg['colors']['white-noise'], label='white-noise', edgecolor='none')
    ax.fill_between(x_detection, continuum_limit, y_detection, color=aspect.cfg['colors']['continuum'], label='continuum', edgecolor='none')

    # Cosmic and single-pixel lines
    ax.fill_betweenx(int_cosmic_ray, 0, res_cosmic_ray, color=aspect.cfg['colors']['cosmic-ray'], label='cosmic-ray', edgecolor='none')

    # Doublet
    ax.fill_between(res_doublet, y_doublet_min, y_doublet_max, color=aspect.cfg['colors']['doublet'], alpha=0.8, label='doublet')

    # Wording
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Gaussian sigma in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})
    ax.legend(loc='lower center', ncol=3, framealpha=0.95, fontsize=10)

    # Axis format
    ax.set_yscale('log')
    ax.set_xlim(0, 3)
    ax.set_ylim(0.01, 10000)

    # Upper axis
    ax2 = ax.twiny()
    ticks_values = ax.get_xticks()
    ticks_labels = [f'{tick:.0f}' for tick in ticks_values*6]
    ax2.set_xticks(ticks_values)  # Set the tick positions
    ax2.set_xticklabels(ticks_labels)
    ax2.set_xlabel(r'$b_{pixels}$ (detection box width in pixels)')

    # Grid
    ax.grid(axis='x', color='0.95', zorder=1)
    ax.grid(axis='y', color='0.95', zorder=1)

    plt.tight_layout()
    plt.savefig(fig_folder/'diagnostic_plot.png')
    plt.show()
