import aspect
import numpy as np
from pathlib import Path

# Configuration
cfg_file = '12_pixels.toml'
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

