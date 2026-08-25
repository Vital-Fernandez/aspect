import numpy as np
import aspect
import lime
import joblib
from pathlib import Path
from matplotlib import pyplot as plt, rc_context
from aspect.tools import detection_function, cosmic_ray_function

# Configuration
cfg_file = '12_pixels_flux.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files
y_arr = np.loadtxt(output_folder/f'pred_array_{norm}_reference_sample.txt')
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_reference_sample.txt', delimiter=',')

box_size = sample_cfg[f'properties_{version}']['box_pixels']
feature_slice = -box_size - 1
lines_arr = data_matrix[:, feature_slice:]

#      5         6          7         8           9           10
#  true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
sn_ratio = data_matrix[:, 0]
res_ratio = data_matrix[:, 1]
true_flux, true_err = data_matrix[:, 5], data_matrix[:, 6]
intg_flux, intg_err = data_matrix[:, 7], data_matrix[:, 8]
gauss_flux, gauss_err = data_matrix[:, 9], data_matrix[:, 10]

# Load the trained model
model_address = '/home/vital/Astrodata/aspect/medium_box/results/aspect_min-max-log_12_pixels_flux_v13_randomforest_model.joblib'
ml_function = joblib.load(model_address)
ml_flux = ml_function.predict(lines_arr)

# Config figure
fig_cfg = lime.theme.fig_defaults(user_fig={
    "figure.figsize": (22, 7),
    "figure.dpi": 100,
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 22,
})

labels = ['Gaussian flux', 'Integrated flux', 'Machine learning flux']
flux_arrays = [gauss_flux, intg_flux, ml_flux]
err_arrays = [gauss_err, intg_err, true_err]

with rc_context(fig_cfg):
    fig, axes = plt.subplots(1, 3, sharey=True)

    detection_range = np.linspace(res_ratio.min(), res_ratio.max(), num=50)

    for idx, (ax, ref, flux_arr, err_arr) in enumerate(zip(axes, labels, flux_arrays, err_arrays)):

        # Convert to log scale for Gaussian and integrated, ML is already in log scale
        if idx != 2:
            flux_diag = np.log10(flux_arr) / 4
        else:
            flux_diag = flux_arr

        # Diagnostic ratio — all three now in log-scaled space
        diag = np.abs(flux_diag / (np.log10(true_flux) / 4) - 1)

        sc = ax.scatter(res_ratio, sn_ratio, c=diag, cmap='viridis', vmin=0, vmax=0.3)
        ax.plot(detection_range, detection_function(detection_range))
        ax.plot(detection_range, cosmic_ray_function(detection_range))

        ax.set_title(ref)
        ax.set_yscale('log')
        ax.set_ylim(4, 100)
        ax.set_xlim(0, 2)

        if idx == 0:
            ax.set_ylabel(r'$\frac{A_{gas}}{\sigma_{noise}}$ (Signal-to-noise)')

        if idx == 2:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(r'$\left|\frac{\log_{10}(F_{measured})/4}{\log_{10}(F_{true})/4} - 1\right|$')

    fig.supxlabel(r'$\frac{\sigma_{gas}}{\Delta\lambda_{inst}} = \sigma_{pixels}$ (Velocity dispersion in pixels)',
                  fontsize=20)
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_folder/f'methodology_comparison_log.png')

