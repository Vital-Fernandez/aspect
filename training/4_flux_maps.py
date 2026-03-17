import numpy as np
import pandas as pd
import aspect
import lime
from pathlib import Path
from matplotlib import pyplot as plt, rc_context

lime.theme.set_style('dark')

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
idcs_emis = ((y_arr == 'emission') | (y_arr == 'cosmic-ray')) & (data_matrix[:, 0] < sn_max) & ~np.isnan(data_matrix[:, 5])
data_matrix = data_matrix[idcs_emis, :]

# Recover the data columns

#      5         6          7         8           9           10
#  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
num_entries = 50
sn_ratio = data_matrix[:, 0]
res_ratio = data_matrix[:, 1]
true_flux = data_matrix[:, 5]
gauss_flux = data_matrix[:, 9]

# true_flux = data_matrix[:, 6]/data_matrix[:, 5]
# intg_flux = data_matrix[:, 8]/data_matrix[:, 7]
# gauss_flux = data_matrix[:, 10]/data_matrix[:, 9]

# true_flux = np.abs(data_matrix[:, 5] - data_matrix[:, 5])/ data_matrix[:, 6]
# intg_flux = np.abs(data_matrix[:, 5] - data_matrix[:, 7])/ data_matrix[:, 8]
# gauss_flux = np.abs(data_matrix[:, 5] - data_matrix[:, 9])/ data_matrix[:, 10]

# gauss_flux = data_matrix[:, 5]
fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (8, 8), "figure.dpi" : 400, "axes.labelsize": 20,
                                            "xtick.labelsize": 16, "ytick.labelsize": 16})
with rc_context(fig_cfg):

    # for idx, (ref, flux_arr) in enumerate(zip(['True flux', 'Integrated flux', 'Gaussian flux'], [true_flux, intg_flux, gauss_flux])):
    for idx, (ref, flux_arr) in enumerate(zip(['True flux', 'Gaussian flux'], [true_flux, gauss_flux])):

        fig, ax = plt.subplots()

        # Get data arrays to generate a matching regular grid
        x_grid = np.linspace(res_ratio.min(), res_ratio.max(), num=num_entries)
        y_grid = np.logspace(np.log10(sn_ratio.min()), np.log10(sn_ratio.max()), num=num_entries)
        x_grid = x_grid[np.abs(x_grid[:, None] - res_ratio).argmin(axis=0)]
        y_grid = y_grid[np.abs(y_grid[:, None] - sn_ratio).argmin(axis=0)]

        # Build a temporary dataframe just for groupby
        df_temp = pd.DataFrame({'x_snapped': x_grid, 'y_snapped': y_grid, 'A': true_flux})
        df_snapped = df_temp.groupby(['x_snapped', 'y_snapped']).agg(A=('A', 'mean')).reset_index()

        #
        sc = ax.scatter(df_snapped['x_snapped'], df_snapped['y_snapped'], c=df_snapped['A'], cmap='viridis')

        # Add a colorbar on the right
        cbar = plt.colorbar(sc)
        cbar.set_label(r'$\frac{\sigma_{line}}{F_{line}}$ (Coefficient of variation)')
        ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

        # ax.set_xlim((0.25, 2))
        # ax.set_ylim((4.5, 10))
        ax.set_yscale('log')
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'{ref}_coeff_var.png')


# # Create scatter plot with colors mapped to 'values'
# sc = plt.scatter(x, y, c=values, cmap='viridis')  # you can try 'plasma', 'coolwarm', etc.
#
# # Add a colorbar on the right
# cbar = plt.colorbar(sc)
# cbar.set_label('Array Value')
#
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Scatter plot colored by array values")
# plt.show()



# import aspect
# import numpy as np
# from pathlib import Path
# from matplotlib import pyplot as plt, rc_context
# import lime
# from scipy.stats import gaussian_kde, binned_statistic_2d
#
# # Convert desired enclosed-probability levels (e.g. 68%, 95%) to density thresholds
# def kde_levels_for_enclosed_probs(pdf2d, probs):
#     """
#     Return strictly increasing density thresholds corresponding to
#     enclosed probabilities in `probs` (e.g. 0.68, 0.95, 0.997).
#     """
#     p = pdf2d.ravel()
#     # Sort densities high->low and build CDF of enclosed mass
#     idx = np.argsort(p)[::-1]
#     p_sorted = p[idx]
#     cdf = np.cumsum(p_sorted)
#     cdf /= cdf[-1]
#
#     # Find the density threshold at each probability
#     raw_levels = []
#     for q in probs:
#         k = np.searchsorted(cdf, q, side="left")
#         k = min(max(k, 0), len(p_sorted)-1)
#         raw_levels.append(p_sorted[k])
#
#     # Matplotlib needs strictly increasing and unique levels
#     levels = np.array(raw_levels, dtype=float)
#     levels = np.unique(levels)           # drop duplicates if any
#     levels.sort()                        # make increasing
#
#     # Keep only the probs that survived de-duplication, for labels
#     # (map density->probability, then rebuild in sorted order)
#     dens_to_prob = {d: pr for d, pr in zip(raw_levels, probs)}
#     probs_sorted = [dens_to_prob[d] for d in levels]
#     return levels, probs_sorted
#
# # Configuration
# cfg_file = '12_pixels.toml'
# sample_cfg = aspect.load_cfg(cfg_file)
# version = sample_cfg['meta']['version']
# norm = sample_cfg['meta']['scale']
# output_folder = Path(sample_cfg['meta']['results_folder'])
#
# # Read the sample files:
# y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
# data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_{version}.txt', delimiter=',')
#
# # Indices emission
# sn_max = 100
# idcs_emis = (y_arr == 'emission') & (data_matrix[:, 0] < sn_max) & ~np.isnan(data_matrix[:, 5])
# data_matrix = data_matrix[idcs_emis, :]
#
# #      5         6          7         8           9           10
# #  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
# # Recover the data columns
# sn_ratio = data_matrix[:, 0]
# res_ratio = data_matrix[:, 1]
# true_flux = data_matrix[:, 6]/data_matrix[:, 5]
# intg_flux = data_matrix[:, 8]/data_matrix[:, 7]
# gauss_flux = data_matrix[:, 10]/data_matrix[:, 9]
#
# nx, ny = 200, 200
# xbins = np.linspace(res_ratio.min(), res_ratio.max(), nx+1)
# ybins = np.linspace(sn_ratio.min(), sn_ratio.max(), ny+1)
# stat, xedges, yedges, _ = binned_statistic_2d(res_ratio, sn_ratio, true_flux, statistic="mean", bins=[xbins, ybins])
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
# xc = 0.5*(xedges[:-1] + xedges[1:])
# yc = 0.5*(yedges[:-1] + yedges[1:])
# XX, YY = np.meshgrid(xc, yc, indexing="xy")
#
# kde = gaussian_kde(np.vstack([res_ratio, sn_ratio]))
# pdf = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(YY.shape)
#
# probs = [0.68, 0.95, 0.997]
# levels, probs_sorted = kde_levels_for_enclosed_probs(pdf, probs)
#
# # Contours: levels must be increasing
# fig, ax = plt.subplots()
# cs = ax.contour(XX, YY, pdf, levels=levels, linewidths=1.5)
# ax.clabel(cs, fmt={lv: f"{p*100:.1f}%" for lv, p in zip(levels, probs_sorted)}, inline=True)
#
# ax.set_xlabel("x"); ax.set_ylabel("y")
# ax.set_title("Value-colored map with KDE isodensity contours")
# plt.tight_layout(); plt.show()