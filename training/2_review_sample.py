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
shape_list = ['white-noise', 'continuum', 'cosmic-ray', 'doublet', 'emission']
sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=6, sample_size=n_points, categories=shape_list)
sample_plotter.show()
