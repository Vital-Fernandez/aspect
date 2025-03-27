import numpy as np

import aspect
from lime.fitting.lines import gaussian_model
from aspect.tools import stratify_sample, detection_function, cosmic_ray_function, broad_component_function

from pathlib import Path
from itertools import product
from tqdm import tqdm

import gc

def store_line(x_arr, y_arr, class_name, i_line, synth_line, x_cord, y_cord, box_size):

    pixel_int = synth_line[idx_0:idx_f]
    min, max, std = np.min(pixel_int), np.max(pixel_int), np.std(pixel_int)

    x_arr[i_line, :5] = y_cord, x_cord, min, max, std
    x_arr[i_line, -box_size:] = synth_line[idx_0:idx_f]
    y_arr[i_line] = class_name

    return i_line + 1

# Load sample
cfg_file = 'medium_box.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
params = sample_cfg[f'properties_{version}']
norm = sample_cfg['meta']['scale']

output_folder = Path(sample_cfg['meta']['results_folder'])
output_folder.mkdir(parents=True, exist_ok=True)

# Grid parameters
n_sigma = params['n_sigma']
err_n_sigma =  params['err_n_sigma']
box_pixels = params['box_pixels']
sigma_pixels = box_pixels/n_sigma

res_ratio_min = params['res_ratio_min']
int_ratio_min = params['int_ratio_min']
int_ratio_max = params['int_ratio_max']
int_ratio_base = params['int_ratio_log_base']

instr_res = params['instr_res']
sample_size = params['int_ratio_points'] * params['res_ratio_points']
half_sample = int(sample_size/2)

# --------- Compute the range of the parameter space
int_ratio_min_log = np.log(int_ratio_min) / np.log(int_ratio_base)
int_ratio_max_log = np.log(int_ratio_max) / np.log(int_ratio_base)
int_ratio_range = np.logspace(int_ratio_min_log, int_ratio_max_log, params['int_ratio_points'], base=int_ratio_max)
res_ratio_range = np.linspace(res_ratio_min, sigma_pixels, params['res_ratio_points'])
combinations = np.array(list(product(int_ratio_range, res_ratio_range)))

print(f'\nInt_ratio size: {params["int_ratio_points"]}')
print(f'Res_ratio size: {params["res_ratio_points"]}')
print(f'combinations : {combinations.size}')

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

doublet_res_min = params['doublet']['min_res_ratio']
doublet_res_max = params['doublet']['max_res_ratio']
doublet_int_min = params['doublet']['min_int_ratio']
doublet_int_max = params['doublet']['max_int_ratio']
doublet_sep_min = params['doublet']['min_separation']
doublet_sep_max = params['doublet']['max_separation']
doublet_int_discr = params['doublet']['discrepancy_factors']

# --------- Containers
print('\nGenerating containers for results')
n_lines = sample_size * 7

# Features  Coordinates    Scale parameters (min, max, std)   + Scale featires + Pixel Features
n_columns = 2              + 3                                + 1              + box_pixels

pred_arr = np.empty(n_lines, dtype='U20')
data_matrix = np.full((n_lines, n_columns), np.nan)

# ---------  Loop through the conditions
print('\nLooping through combinations')
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")

counter = 0
for idx, (int_ratio, res_ratio) in enumerate(bar):

    # Continuum components
    cont_arr = gradient_arr[idx] * wave_arr + cont_level
    white_noise_arr = normal_noise_matrix[idx, :]
    noise_i = uniform_noise_arr[idx]

    # Line components
    amp = int_ratio * noise_i
    sigma = res_ratio * instr_res
    line_pixels = sigma * n_sigma
    theo_flux = amp * 2.5066282746 * sigma
    true_error = noise_i * np.sqrt(2 * err_n_sigma * instr_res * sigma)

    # Reference values
    detection_value = detection_function(res_ratio)
    cosmic_ray_res = cosmic_ray_function(int_ratio, res_ratio_check=False)

    # Detection cases
    if int_ratio >= detection_value:

        # Flux array
        flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr

        # Line
        if res_ratio > cosmic_ray_res:
            shape = 'emission'

        # Single pixel
        else:
            shape = 'cosmic-ray'

        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)

        # Absorption:
        flux_arr = gaussian_model(wave_arr, -amp, mu_line, sigma) + white_noise_arr + cont_arr

        if res_ratio > cosmic_ray_res:
            shape = 'absorption'
        else:
            shape = 'dead-pixel'

        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)

    # Continuum
    else:

        # White noise
        if (int_ratio >= white_noise_min_int_ratio) and (int_ratio <= white_noise_max_int_ratio):
            shape = 'white-noise'
            white_noise_arr = np.random.normal(loc=0, scale=1/int_ratio, size=params['uncrop_array_size'])
            flux_arr = white_noise_arr + cont_arr

        # Continuum
        else:
            shape = 'continuum'
            flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr

        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)

    # Doublet
    if (int_ratio > doublet_int_min) & (int_ratio < doublet_int_max) & (res_ratio > doublet_res_min) & (res_ratio < doublet_res_max):

        # Compute the doublet
        sep = np.random.uniform(doublet_sep_min, doublet_sep_max)
        int_diff = np.random.uniform(doublet_int_discr[0], doublet_int_discr[1])
        amp1, amp2 = amp, amp * int_diff
        mu1, mu2 = mu_line - sep, mu_line + sep
        sigma1, sigma2 = sigma, sigma * 1

        # Generate the profiles
        gauss1 = gaussian_model(wave_arr, amp1, mu1, sigma1)
        gauss2 = gaussian_model(wave_arr, amp2, mu2, sigma2)
        flux_arr = gauss1 + gauss2 + white_noise_arr + cont_arr

        # Store the data
        counter = store_line(data_matrix, pred_arr, 'doublet', counter, flux_arr, res_ratio, int_ratio, box_pixels)

# ---------  Save the results
idcs_empty = pred_arr != ''
data_matrix = data_matrix[idcs_empty, :]
pred_arr = pred_arr[idcs_empty]

# Crop the dataset to use the same number of points as the smallest number of points
data_matrix, pred_arr = stratify_sample(data_matrix, pred_arr, randomize=True)

# Two individual files per sample
np.savetxt(output_folder/f'data_array_{version}.txt', data_matrix, fmt='%.6f', delimiter=',')
np.savetxt(output_folder/f'pred_array_{version}.txt', pred_arr, fmt='%s')

# Clear the memory just in case
del data_matrix
del pred_arr
gc.collect()

# Load and save a scaled version
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
aspect.tools.scale_min_max(data_matrix, box_pixels, axis=1)
np.savetxt(output_folder/f'data_array_{norm}_{version}.txt', data_matrix, fmt='%.6f', delimiter=',')