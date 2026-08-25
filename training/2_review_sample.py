import numpy as np
import aspect
from pathlib import Path
from matplotlib import pyplot as plt, colors

# Configuration
cfg_file = '12_pixels_flux.toml' # '24_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])
science_type = sample_cfg['meta'].get('target_science')

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{norm}_{version}.txt')
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_{version}.txt', delimiter=',')

# Classifier
if science_type != 'line_flux':
    n_points = 5000
    shape_list = ['cosmic-ray', 'white-noise', 'continuum', 'emission', 'doublet-em']
    sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=12, sample_size=n_points, categories=shape_list)
    sample_plotter.show()

# Flux measurement
else:


    # Scatter plot
    fig, ax = plt.subplots()

    idcs_limit = 5000
    ycoords, xcoords, fluxoords = data_matrix[:,0][:idcs_limit], data_matrix[:,1][:idcs_limit], y_arr[:idcs_limit]
    # fluxoords = np.log10(data_matrix[:, 5][:idcs_limit])/4 - fluxoords

    # sc = ax.scatter(xcoords, ycoords, c=fluxoords, cmap='magma', norm=colors.LogNorm())
    sc = ax.scatter(xcoords, ycoords, c=fluxoords, cmap='magma')
    cbar = plt.colorbar(sc)
    ax.set_yscale('log')
    # ax.set_ylim(4, 100)
    # ax.set_xlim(0, 2)
    plt.tight_layout()
    plt.show()

