import aspect
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt, rc_context
import lime

# Configuration
cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_{version}.txt', delimiter=',')

# Indices emission
sn_max = 100
idcs_emis = (y_arr == 'emission') & (data_matrix[:, 0] < sn_max) & ~np.isnan(data_matrix[:, 5])
data_matrix = data_matrix[idcs_emis, :]

# Recover the data columns
#      5         6          7         8           9           10
#  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
sn_ratio = data_matrix[:, 0]
res_ratio = data_matrix[:, 1]
true_flux = data_matrix[:, 6]/data_matrix[:, 5]
intg_flux = data_matrix[:, 8]/data_matrix[:, 7]
gauss_flux = data_matrix[:, 10]/data_matrix[:, 9]

fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (5, 5)})
num_entries = 5000
with rc_context(fig_cfg):
    fig, ax = plt.subplots()

    idcs = (intg_flux > 0) & (intg_flux < 1.00) & ~np.isnan(intg_flux) & (gauss_flux > 0) & (gauss_flux < 1.00) & ~np.isnan(gauss_flux)

    intg_arr = (data_matrix[:, 7][idcs] - data_matrix[:, 5][idcs])/data_matrix[:, 5][idcs]
    gauss_arr = (data_matrix[:, 9][idcs] - data_matrix[:, 5][idcs])/data_matrix[:, 5][idcs]
    true_arr = data_matrix[:, 5]

    ax.scatter(true_arr[:num_entries], intg_arr[:num_entries], alpha=0.2, color='tab:blue')
    ax.scatter(true_arr[:num_entries], gauss_arr[:num_entries], alpha=0.2, color='tab:orange')

    ax.legend()
    plt.show()


# fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (5, 5)})
# with rc_context(fig_cfg):
#     fig, ax = plt.subplots()
#
#     idcs = (intg_flux > 0) & (intg_flux < 1.00) & ~np.isnan(intg_flux) & (gauss_flux > 0) & (gauss_flux < 1.00) & ~np.isnan(gauss_flux)
#     arr = (data_matrix[:, 7][idcs] - data_matrix[:, 5][idcs])/data_matrix[:, 5][idcs]
#     ax.hist(arr, bins=20, label=f'Integrated ({arr.shape})', density=True,
#             histtype = 'step', hatch = '/', edgecolor = 'orange')
#
#     arr = (data_matrix[:, 9][idcs] - data_matrix[:, 5][idcs])/data_matrix[:, 5][idcs]
#     ax.hist(arr, bins=20, label=f'Gaussian ({arr.shape})', density=True,
#             histtype='step', hatch='o', edgecolor='green')
#
#     ax.legend()
#     plt.show()


# fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (5, 5)})
# with rc_context(fig_cfg):
#     fig, ax = plt.subplots()
#
#     ax.hist(true_flux, bins=20, label=f'True normalized scatter ({np.sum(~np.isnan(true_flux))})', density=True,
#             histtype='step', facecolor='blue', edgecolor='none', alpha=0.5, fill=True)
#
#     idcs = (intg_flux > 0) & (intg_flux < 1.00)
#     ax.hist(intg_flux[idcs], bins=20, label=f'Intg normalized scatter ({np.sum(~np.isnan(intg_flux[idcs]))})', density=True,
#             histtype = 'step', hatch = '/', edgecolor = 'orange')
#
#     idcs = (gauss_flux > 0) & (gauss_flux < 1.00)
#     ax.hist(gauss_flux[idcs], bins=20, label=f'Gauss normalized scatter ({np.sum(~np.isnan(gauss_flux[idcs]))})', density=True,
#             histtype='step', hatch='o', edgecolor='green')
#
#     ax.legend()
#     plt.show()
