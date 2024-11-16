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
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')

# Normalization
data_matrix[:, 2:] = aspect.workflow.feature_scaling(data_matrix[:, 2:], 'min-max')

# Save the results
np.savetxt(output_folder/f'data_array_{norm}_{version}.txt', data_matrix, fmt='%.6f', delimiter=',')
