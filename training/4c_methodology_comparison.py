import numpy as np
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
data_matrix = np.loadtxt('/home/vital/Astrodata/aspect/medium_box/data_array_min-max_reference_sample.txt', delimiter=',')

# x_arr.shape
# Out[4]: (1375290, 24)
data_matrix = data_matrix[::3]

# Input model data (spectral features plus intensity)
box_size = sample_cfg[f'randomforest_flux_{version}']['box_size']
feature_slice = -box_size - 1
x_sample = data_matrix[:, feature_slice:]

#      5         6          7         8           9           10
#  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
sn_ratio = data_matrix[:, 0]
res_ratio = data_matrix[:, 1]
true_flux, true_err = data_matrix[:, 5], data_matrix[:, 6]
intg_flux, intg_err = data_matrix[:, 7], data_matrix[:, 8]
gauss_flux, gauss_err = data_matrix[:, 9], data_matrix[:, 10]

# Load the trained model
model_address = '/home/vital/Astrodata/aspect/medium_box/results/aspect_min-max_12_pixels_v11_MLP_flux_model.joblib'
ml_function = joblib.load(model_address)
ml_flux = np.power(10, ml_function.predict(x_sample))

# Config figure
fig_cfg = lime.theme.fig_defaults(user_fig={"figure.figsize" : (8, 8), "figure.dpi" : 400, "axes.labelsize": 30,
                                            "xtick.labelsize": 16, "ytick.labelsize": 16, "axes.titlesize": 40,})
with rc_context(fig_cfg):

    for idx, (ref, flux_arr, err_arr) in enumerate(zip(['True flux', 'Gaussian flux', 'Integrated flux', 'ML flux'],
                                                       [true_flux, gauss_flux, intg_flux, ml_flux],
                                                       [true_err, gauss_err, intg_err, true_err])):

        # Diagnostic ratio
        # diag = err_arr/flux_arr
        diag = np.abs(flux_arr/true_flux - 1)

        fig, ax = plt.subplots()

        sc = ax.scatter(res_ratio, sn_ratio, c=diag, cmap='viridis', vmin=0, vmax=1)

        detection_range = np.linspace(res_ratio.min(), res_ratio.max(), num=50)
        ax.plot(detection_range, detection_function(detection_range))
        ax.plot(detection_range, cosmic_ray_function(detection_range))

        # Add a colorbar on the right
        ax.set_title(ref)
        cbar = plt.colorbar(sc,)
        cbar.set_label(r'$\frac{F_{measured}}{F_{true}} - 1$')
        ax.update({'xlabel': r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
                   'ylabel': r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)'})

        ax.set_yscale('log')
        ax.set_ylim(4, 50)
        ax.set_xlim(0, 2)
        plt.tight_layout()
        print(output_folder)
        plt.savefig(output_folder/f'{ref}_method_accuracy.png')
        # plt.show()
