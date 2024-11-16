import numpy as np

import aspect
from lime.model import gaussian_model
from lime.recognition import detection_function, cosmic_ray_function, broad_component_function
from aspect.tools import stratify_sample

from pathlib import Path
from itertools import product
from tqdm import tqdm


def store_line(x_arr, y_arr, class_name, i_line, synth_line, x_cord, y_cord):

    y_arr[i_line] = class_name
    x_arr[i_line, :2] = y_cord, x_cord
    x_arr[i_line, 2:] = synth_line[idx_0:idx_f]

    return i_line + 1

cfg_file = 'medium_box.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
output_folder = Path(sample_cfg['meta']['results_folder'])
output_folder.mkdir(parents=True, exist_ok=True)
params = sample_cfg[f'properties_{version}']

# Grid parameters
n_sigma = params['n_sigma']
err_n_sigma =  params['err_n_sigma']
box_pixels = params['box_pixels']
sigma_pixels = box_pixels/n_sigma

# res_ratio_min = params['res_ratio_min']
# int_ratio_min = params['int_ratio_min']
# int_ratio_max = params['int_ratio_max']
# int_ratio_base = params['int_ratio_log_base']

instr_res = params['instr_res']

res_points = 50
sep_points = 50
sample_size = sep_points * res_points
half_sample = int(sample_size/2)

# --------- Compute the range of the parameter space
# int_ratio_min_log = np.log(int_ratio_min) / np.log(int_ratio_base)
# int_ratio_max_log = np.log(int_ratio_max) / np.log(int_ratio_base)
# int_ratio_range = np.logspace(int_ratio_min_log, int_ratio_max_log, params['int_ratio_points'], base=int_ratio_max)
res_ratio_range = np.linspace(1, 1.6, res_points)
separation_range = np.linspace(1.20, 3, sep_points)
combinations = np.array(list(product(separation_range, res_ratio_range)))

# --------- Continuum
print('\nGenerating continuum parameters')

# Inclination
cont_level = params['cont_level']
angle_min = params['angle_min']
angle_max = params['angle_max']
gradient_arr = np.tan(np.deg2rad(np.random.uniform(angle_min, angle_max, sample_size))) #np.full(sample_size, np.tan(np.deg2rad(0)))

# Wavelength values
mu_line = params['mu_line']
wave_arr = np.arange(- params['uncrop_array_size'] / 2, params['uncrop_array_size'] / 2, instr_res)
idx_zero = np.searchsorted(wave_arr, mu_line)
idx_0, idx_f = int(idx_zero - box_pixels/2), int(idx_zero + box_pixels/2)

# Generate the random noise
uniform_noise_arr = np.full((sample_size,1), 1)
normal_noise_matrix = np.random.normal(loc=0, scale=uniform_noise_arr, size=(sample_size, params['uncrop_array_size']))

# --------- Features
cr_boundary = params['cosmic-ray']['cosmic_ray_boundary']
white_noise_min_int_ratio = params['white_noise']['min_int_ratio']
white_noise_max_int_ratio = params['white_noise']['max_int_ratio']

# --------- Containers
print('\nGenerating containers for results')
n_lines = sample_size * 1
pred_arr = np.empty(n_lines, dtype='U20')
data_matrix = np.full((n_lines, box_pixels + 2), np.nan)

# ---------  Loop through the conditions
print('\nLooping through combinations')
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")

counter = 0
int_ratio = 20
for idx, (sep, res_ratio) in enumerate(bar):

    # Continuum components
    cont_arr = 0 * wave_arr + cont_level
    white_noise_arr = normal_noise_matrix[idx, :]
    noise_i = uniform_noise_arr[idx]

    # Line components
    amp = int_ratio * noise_i
    sigma = res_ratio * instr_res
    line_pixels = sigma * n_sigma
    theo_flux = amp * 2.5066282746 * sigma
    true_error = noise_i * np.sqrt(2 * err_n_sigma * instr_res * sigma)

    # Compute the doublet
    sigma1, sigma2 = sigma, sigma * 1
    amp1, amp2 = amp, amp * 1
    mu1, mu2 = mu_line - sep, mu_line + sep

    # Generate the profiles
    gauss1 = gaussian_model(wave_arr, amp1, mu1, sigma1)
    gauss2 = gaussian_model(wave_arr, amp2, mu2, sigma2)
    flux_arr = gauss1 + gauss2 + white_noise_arr + cont_arr

    # Store the data
    counter = store_line(data_matrix, pred_arr, 'doublet', counter, flux_arr, res_ratio, sep)

    # # Detection cases
    # if int_ratio >= detection_value:
    #
    #     # Flux array
    #     flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr
    #
    #     # Line
    #     if res_ratio > cosmic_ray_res:
    #         shape = 'emission'
    #
    #     # Single pixel
    #     else:
    #         if int_ratio > cr_boundary: # Cosmic ray
    #             shape = 'cosmic-ray'
    #
    #         else: # Pixel line
    #             shape = 'pixel-line'
    #
    #     counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio)
    #
    #     # Absorption:
    #     flux_arr = gaussian_model(wave_arr, -amp, mu_line, sigma) + white_noise_arr + cont_arr
    #
    #     if res_ratio > cosmic_ray_res:
    #         shape = 'absorption'
    #     else:
    #         shape = 'dead-pixel'
    #
    #     counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio)
    #
    # # Continuum
    # else:
    #
    #     # White noise
    #     if (int_ratio >= white_noise_min_int_ratio) and (int_ratio <= white_noise_max_int_ratio):
    #         shape = 'white-noise'
    #         white_noise_arr = np.random.normal(loc=0, scale=1/int_ratio, size=params['uncrop_array_size'])
    #         flux_arr = white_noise_arr + cont_arr
    #
    #     # Continuum
    #     else:
    #         shape = 'continuum'
    #         flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr
    #
    #     counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio)


# ---------  Save the results

# Two individual files per sample
np.savetxt(output_folder/f'data_array_doublet.txt', data_matrix, fmt='%.6f', delimiter=',')
np.savetxt(output_folder/f'pred_array_doublet.txt', pred_arr, fmt='%s')
