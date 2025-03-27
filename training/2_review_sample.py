import numpy as np
import aspect
from pathlib import Path


# Configuration
cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')

# Plot sample
n_points = 5000
# shape_list = ['white-noise', 'continuum', 'emission', 'cosmic-ray']
shape_list = ['white-noise', 'continuum', 'emission', 'cosmic-ray', 'doublet']
# shape_list = ['white-noise', 'continuum', 'Hbeta_OIII-doublet']
sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=6, sample_size=n_points, categories=shape_list)
sample_plotter.show()


# # Read the sample files:
# y_arr = np.loadtxt(output_folder/f'pred_array_doublet.txt', dtype=str)
# data_matrix = np.loadtxt(output_folder/f'data_array_doublet.txt', delimiter=',')
#
# # Plot sample
# n_points = 2500
# shape_list = ['doublet']
# sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, sample_size=n_points, categories=shape_list, dtype='doublet')
# sample_plotter.show()
