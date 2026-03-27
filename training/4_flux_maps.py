import numpy as np
import pandas as pd
import aspect
import lime
import joblib
from pathlib import Path
from matplotlib import pyplot as plt, rc_context
from aspect.tools import detection_function, cosmic_ray_function
lime.theme.set_style('dark')

# Configuration
cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_reference_sample.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_reference_sample.txt', delimiter=',')


# x_arr.shape
# Out[4]: (1375290, 24)
data_matrix = data_matrix[::3]

#      5         6          7         8           9           10
#  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
sn_ratio = data_matrix[:, 0]
res_ratio = data_matrix[:, 1]
true_flux, true_err = data_matrix[:, 5], data_matrix[:, 6]
intg_flux, intg_err = data_matrix[:, 7], data_matrix[:, 8]
gauss_flux, gauss_err = data_matrix[:, 9], data_matrix[:, 10]

# Input model data (spectral features plus intensity)
box_size = sample_cfg[f'randomforest_flux_{version}']['box_size']
feature_slice = -box_size - 1
x_sample = data_matrix[:, feature_slice:]

# gauss_flux = data_matrix[:, 5]

# Accuracy maps
fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (8, 8), "figure.dpi" : 400, "axes.labelsize": 20,
                                            "xtick.labelsize": 16, "ytick.labelsize": 16})
with rc_context(fig_cfg):

    for idx, (ref, flux_arr, err_arr) in enumerate(zip(['True flux', 'Gaussian flux', 'Integrated flux'],
                                                       [true_flux, gauss_flux, intg_flux],
                                                       [true_err, true_err, true_err])):

        fig, ax = plt.subplots()

        sc = ax.scatter(res_ratio, sn_ratio, c=err_arr/flux_arr, cmap='viridis', vmin=0, vmax=0.6)

        # Add a colorbar on the right
        cbar = plt.colorbar(sc,)
        cbar.set_label(r'$\frac{\sigma_{line}}{F_{line}}$ (Coefficient of variation)')
        ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

        # ax.set_yscale('log')
        # plt.tight_layout()
        # plt.show()

        detection_range = np.linspace(res_ratio.min(), res_ratio.max(), num=50)
        ax.plot(detection_range, detection_function(detection_range))
        ax.plot(detection_range, cosmic_ray_function(detection_range))

        ax.set_yscale('log')
        ax.set_ylim(4, 100)
        ax.set_xlim(0, 2)
        plt.tight_layout()
        plt.show()


# Load the trained model
model_address = Path(output_folder)/'results'/f'aspect_{norm}_{version}_flux_model.joblib'
ml_function = joblib.load(model_address)
ml_arr = np.power(10, ml_function.predict(x_sample))
relative_error = (np.abs(ml_arr - true_flux)/true_flux) * 100


# idcs_max = np.where(relative_error > 50)[0]
# for i, idx in enumerate(idcs_max):
#     fig, ax = plt.subplots()
#     AI_i = np.power(10, ml_function.predict(np.atleast_2d(x_sample[idx, :]))[0])
#
#     label = f'AI_i = {AI_i:0.3f}, SN = {res_ratio[idx]:0.3f}, sig_pix = {sn_ratio[idx]:0.3f}'
#     title = (f'({i}/{len(idcs_max)}): True = {true_flux[idx]:0.3f}, AI = {ml_arr[idx]:0.3f},  '
#              f'Gauss = {gauss_flux[idx]:0.3f}, Intg = {intg_flux[idx]:0.3f}')
#
#     ax.step(np.arange(12), x_sample[idx, 1:], label=label)
#     ax.set_title(title)
#     ax.legend()
#     plt.show()

# Accuracy maps
fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (8, 8), "figure.dpi" : 400,
                                            "axes.labelsize": 20, "xtick.labelsize": 16,
                                            "ytick.labelsize": 16})
with rc_context(fig_cfg):

    fig, ax = plt.subplots()

    sc = ax.scatter(res_ratio, sn_ratio, c=relative_error, cmap='viridis', vmin=0, vmax=100)

    detection_range = np.linspace(res_ratio.min(), res_ratio.max(), num=50)
    ax.plot(detection_range, detection_function(detection_range))
    ax.plot(detection_range, cosmic_ray_function(detection_range))

    # Add a colorbar on the right
    cbar = plt.colorbar(sc,)
    cbar.set_label(r'Prediction difference (%)')
    ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
               'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

    ax.set_yscale('log')
    ax.set_ylim(4, 100)
    ax.set_xlim(0, 2)
    plt.tight_layout()
    plt.show()