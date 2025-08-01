import numpy as np
from matplotlib import pyplot as plt, rc_context
import matplotlib.gridspec as gridspec

import aspect
from pathlib import Path
import lime

# Configuration
cfg_file = '../../training/12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])
categories = sample_cfg[f'properties_{version}']['categories']
fig_folder = Path(f'/home/vital/Dropbox/Astrophysics/Tools/aspect')

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
data_matrix = data_matrix[:, 6:]
x_range = np.arange(12)

target_idcs = {'white-noise':116,
               'continuum':128,#/128,
               'emission':103,
               'doublet':110,
               'absorption': 36,
               'cosmic-ray': 153,
               'dead-pixel':61,}


fig_cfg = {"figure.dpi" : 350,
           "figure.figsize" : (12, 2),
           'axes.titlesize': 40,
           "axes.labelsize": 16,
           "xtick.labelsize": 14,
           'font.family': 'Times New Roman',
           "text.usetex": True,
           }

conf = lime.theme.fig_defaults(fig_cfg)

with rc_context(conf):

    # Create figure and a 2x4 grid
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 4, figure=fig)


    # Top row: P1–P3 (columns 0–2), P4 spans rows 0–1 in column 3
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))  # P1
    axes.append(fig.add_subplot(gs[0, 1]))  # P2
    axes.append(fig.add_subplot(gs[0, 2]))  # P3
    axes.append(fig.add_subplot(gs[:, 3]))  # P4 spans both rows in last column
    axes.append(fig.add_subplot(gs[1, 0]))  # P5
    axes.append(fig.add_subplot(gs[1, 1]))  # P6
    axes.append(fig.add_subplot(gs[1, 2]))  # P7

    # Plot in the first 7 axes
    for i, item in enumerate(target_idcs.items()):
        comp, idx = item
        y_arr = data_matrix[idx, :]
        axes[i].step(x_range, y_arr, where='mid', color=aspect.cfg['colors'][comp], linewidth=3)
        axes[i].set_title(f"{comp.capitalize()}")

        axes[i].set_ylabel('')
        axes[i].set_yticks([])  # Remove y-axis ticks
        axes[i].set_yticklabels([])
        axes[i].set_xlabel('')
        axes[i].set_xticks([])  # Remove y-axis ticks
        axes[i].set_xticklabels([])


    plt.tight_layout()
    # plt.show()
    plt.savefig(fig_folder/'categories_example.png')



#     # Create 4 rows x 2 columns = 8 subplots
#     fig, axes = plt.subplots(2, 4)
#     axes = axes.flatten()  # Flatten to 1D array for easy indexing
#
#     # Plot in the first 7 axes
#     for i, item in enumerate(target_idcs.items()):
#         comp, idx = item
#         y_arr = data_matrix[idx, :]
#         axes[i].step(x_range, y_arr, where='mid', color=aspect.cfg['colors'][comp])
#         axes[i].set_title(f"{comp.capitalize()}")
#
#         axes[i].set_ylabel('')
#         axes[i].set_yticks([])  # Remove y-axis ticks
#         axes[i].set_yticklabels([])
#         axes[i].set_xlabel('')
#         axes[i].set_xticks([])  # Remove y-axis ticks
#         axes[i].set_xticklabels([])
#
#     # Hide the 8th (unused) axis
#     axes[7].axis('off')
#
#     plt.tight_layout()
#     plt.show()

# undefined = 'black'
# white-noise = '#C41E3A'         # Red
# continuum = '#F48CBA'           # Pink
# emission = '#00FF98'            # Spring Green
# cosmic-ray= '#FFF468'           # Yellow
# pixel-line = '#0070DD'          # Blue
# broad = '#A330C9'               # Dark magenta
# doublet = '#3FC7EB'             # Light blue
# peak = '#C69B6D'                # Tan
# absorption = '#FF7C0A'          # Orange
# dead-pixel = '#8788EE'          # Purple
# phl293B = "#33937F"             # Dark Emerald
# Hbeta_OIII-doublet = "#33937F"  # Dark Emerald


# target = 'absorption'
# for idx_feature in np.where(y_arr == target)[0]:
#     print(idx_feature)
#     fig, ax = plt.subplots()
#     ax.step(x_range, data_matrix[idx_feature, :], label=y_arr[idx_feature], color=aspect.cfg['colors'][y_arr[idx_feature]], where='mid')
#     plt.show()

