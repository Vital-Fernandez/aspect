import numpy as np
from matplotlib import pyplot as plt, rc_context
import matplotlib.gridspec as gridspec

import aspect
from pathlib import Path
import lime


# Configuration
version = '12_pixels_v8'
output_folder = Path('/home/vital/Astrodata/aspect/medium_box/PorSi/')
fig_folder = Path(f'/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
data_matrix = data_matrix[:, -12:]
x_range = np.arange(12)

target_idcs = {'white-noise':116,
               'cosmic-ray': 1,
               'emission':81,
               'doublet-em':46,
               'continuum':32,
               'dead-pixel':34,
               'absorption': 109,
               'doublet-abs':129}

fig_cfg = {"figure.dpi" : 350,
           "figure.figsize" : (12, 2),
           'axes.titlesize': 40,
           "axes.labelsize": 16,
           "xtick.labelsize": 14,
           'font.family': 'Times New Roman',
           "text.usetex": True,
           }

conf = lime.theme.fig_defaults(fig_cfg)
lime.theme.set_style('dark')
with rc_context(conf):

    # Create figure and a 2x4 grid
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # Top row: P1–P3 (columns 0–2), P4 spans rows 0–1 in column 3
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # P1
    axes.append(fig.add_subplot(gs[0, 1]))  # P2
    axes.append(fig.add_subplot(gs[0, 2]))  # P3
    axes.append(fig.add_subplot(gs[0, 3]))  # P4
    axes.append(fig.add_subplot(gs[1, 0]))  # P5
    axes.append(fig.add_subplot(gs[1, 1]))  # P6
    axes.append(fig.add_subplot(gs[1, 2]))  # P7
    axes.append(fig.add_subplot(gs[1, 3]))  # P8

    # Plot in the first 7 axes
    for i, item in enumerate(target_idcs.items()):
        comp, idx = item
        y_arr = data_matrix[idx, :]
        axes[i].step(x_range, y_arr, where='mid', color=aspect.cfg['colors'][comp], linewidth=3)
        axes[i].set_title(f"{comp.capitalize()}")

        axes[i].set_ylabel('')
        axes[i].set_yticks([])
        axes[i].set_yticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])
        axes[i].set_xticklabels([])

    plt.tight_layout()
    plt.show()