import aspect
import numpy as np
from pathlib import Path

# Configuration
cfg_file = '12_pixels.toml'
flux_version = False
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
algorithm_label = sample_cfg['meta']['algorithm']
output_folder = Path(sample_cfg['meta']['results_folder'])

# Read the sample files:
category_path = output_folder / f'pred_array_{version}.txt'
data_matrix_path = output_folder/f'data_array_{norm}_{version}.txt'
category_arr = np.loadtxt(category_path, dtype=str)
data_matrix = np.loadtxt(data_matrix_path, delimiter=',')

# Training the sample
if flux_version is False:
    label = f'aspect_{norm}_{version}_{algorithm_label}_model'
    conf_version = f'{algorithm_label}_{version}'

    cfg = sample_cfg[conf_version]
    cfg['scale'] = sample_cfg['meta']['scale']
    cfg['box_size'] = sample_cfg[f'properties_{version}']['box_pixels']
    aspect.trainer.components_trainer(label, data_matrix, category_arr, cfg, None, output_folder=output_folder)

else:

    # Select the categories with flux
    flux_arr = data_matrix[:, 5]  # True flux
    idcs_emission = (category_arr == 'emission') | (category_arr == 'cosmic-ray')
    flux_arr = flux_arr[idcs_emission]
    data_matrix = data_matrix[idcs_emission, :]

    # Training the sample
    label = f'aspect_{norm}_{version}_{algorithm_label}_flux_model'
    conf_version = f'{algorithm_label}_flux_{version}'

    cfg = sample_cfg[conf_version]
    cfg['scale'] = sample_cfg['meta']['scale']
    cfg['box_size'] = sample_cfg[f'properties_{version}']['box_pixels']
    aspect.trainer.components_trainer(label, data_matrix, flux_arr, cfg, None,
                                      output_folder=output_folder, classification=False)

