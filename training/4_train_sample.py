import numpy as np
import aspect
from pathlib import Path

# Configuration
cfg_file = 'medium_box.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_{version}.txt', delimiter=',')

# Training the sample
label = f'aspect_{norm}_{version}_model'
cfg = sample_cfg[f'randomforest_{version}']
cfg['scale'] = sample_cfg['meta']['scale']
cfg['box_size'] = sample_cfg[f'properties_{version}']['box_pixels']
aspect.trainer.components_trainer(label, data_matrix, y_arr, cfg, None, output_folder=output_folder)

# # Read the sample files:
# y_arr = np.loadtxt(output_folder/f'pred_array_doublet.txt', dtype=str)
# data_matrix = np.loadtxt(output_folder/f'data_array_doublet.txt', delimiter=',')
#
# # Plot sample
# n_points = 2500
# shape_list = ['doublet']
# sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, sample_size=n_points, categories=shape_list, dtype='doublet')
# sample_plotter.show()
