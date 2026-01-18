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
shape_list = ['emission', 'cosmic-ray']
sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=12, sample_size=n_points, categories=shape_list)
sample_plotter.show()



# # Min - max log
# min_arr = data_matrix[:, 2]
# max_arr = data_matrix[:, 3]
# std_arr = data_matrix[:, 4]
# features_arr= data_matrix[:, -12:]
# ratio_arr = data_matrix[:, -13]
#
# # Plot sample
# n_points = 5000
# shape_list = ['white-noise', 'continuum', 'cosmic-ray', 'emission']
# sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=12, sample_size=n_points, categories=shape_list,
#                                           color_array=ratio_arr)
# sample_plotter.show()

# # Configuration
# cfg_file = '12_pixels.toml'
# sample_cfg = aspect.load_cfg(cfg_file)
# version = sample_cfg['meta']['version']
# norm = sample_cfg['meta']['scale']
# output_folder = Path(sample_cfg['meta']['results_folder'])
#
# # Read the sample files:
# y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
# data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
#
# # Min - max log
# min_arr = data_matrix[:, 2]
# max_arr = data_matrix[:, 3]
# std_arr = data_matrix[:, 4]
# features_arr= data_matrix[:, -12:]
# scale_feature = data_matrix[:, -13]
#
# ratio_arr = np.log10(std_arr)/4 # STD
# ratio_arr = np.log10(max_arr-min_arr)/4 # MAX-MIN
# # ratio_arr = np.log10(np.median(features_arr, axis=1)-min_arr)/4 # MAD
# # ratio_arr = np.log10(np.sqrt(np.mean(np.square(features_arr), axis=1)))/4 # RMS
# ratio_arr = scale_feature
#
# # Plot sample
# n_points = 5000
# shape_list = ['white-noise', 'continuum', 'cosmic-ray', 'emission']
# sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, idx_features=12, sample_size=n_points, categories=shape_list,
#                                           color_array=ratio_arr)
# sample_plotter.show()

