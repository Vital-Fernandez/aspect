import numpy as np

import aspect
import lime
from lime.fitting.lines import gaussian_model
from aspect.tools import stratify_sample, detection_function, cosmic_ray_function, doublet_model

from pathlib import Path
from itertools import product
from tqdm import tqdm
import gc


# Change Hbeta bands
lime.lineDB.frame.loc['H1_4861A', ['w1', 'w2', 'w3', 'w4', 'w5', 'w6']] = 4800, 4820, 4840, 4880, 4900, 4920


def line_fitting(synth_arr, class_name, amp, sigma, noise_arr, wave_arr, instr_resolution=1):

    # Run the fit
    spec = lime.Spectrum(wave_arr + 4861.250000, synth_arr, input_err=None, redshift=0, norm_flux=1)
    spec.fit.bands('H1_4861A', cont_source='adjacent', err_from_bands=True)

    # Save the measurements
    true_flux = amp * 2.5066282746 * sigma
    true_err = 2 * spec.frame.at['H1_4861A', 'cont_err'] * np.sqrt(2 * sigma * 1)
    intg_flux, intg_err = spec.frame.at['H1_4861A', 'intg_flux'], spec.frame.at['H1_4861A', 'intg_flux_err']
    profile_flux, profile_err = spec.frame.at['H1_4861A', 'profile_flux'], spec.frame.at['H1_4861A', 'profile_flux_err']

    # if class_name != 'emission':
    #     return np.nan,  np.nan,  np.nan,  np.nan,  np.nan, np.nan
    # else:
    #     if (amp >= 3) and (amp <= 50):
    #
    #         # Run the fit
    #         spec = lime.Spectrum(wave_arr + 4861.250000, synth_arr, input_err=None, redshift=0, norm_flux=1)
    #         spec.fit.bands('H1_4861A', cont_source='adjacent', err_from_bands=True)
    #
    #         # Save the measurements
    #         true_flux = amp * 2.5066282746 * sigma
    #         true_err = 2 * spec.frame.at['H1_4861A', 'cont_err'] * np.sqrt(2 * sigma * 1)
    #         intg_flux, intg_err = spec.frame.at['H1_4861A', 'intg_flux'], spec.frame.at['H1_4861A', 'intg_flux_err']
    #         profile_flux, profile_err = spec.frame.at['H1_4861A', 'profile_flux'], spec.frame.at['H1_4861A', 'profile_flux_err']
    #
    #         return true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
    #
    #     else:
    #         return np.nan,  np.nan, np.nan,  np.nan,  np.nan, np.nan

    return true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err




def store_line(x_arr, y_arr, class_name, i_line, synth_line, x_cord, y_cord, box_size, amp_i=np.nan, sigma_i=np.nan,
               noise_arr_i=np.nan, wave_arr_i=np.nan, include_fit=False):

    pixel_int = synth_line[idx_0:idx_f]
    min, max, std = np.min(pixel_int), np.max(pixel_int), np.std(pixel_int)

    # Compute the flux for the emission lines
    if include_fit:
        true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err = line_fitting(synth_line, class_name,
                                                                                           amp_i, sigma_i,
                                                                                           noise_arr_i, wave_arr_i)
    else:
        true_flux, true_err = amp * 2.5066282746 * sigma, np.nan
        intg_flux, intg_err, profile_flux, profile_err = np.nan, np.nan, np.nan, np.nan

    # Store the parameters  0       1     2    3    4       5         6          7         8           9            10
    x_arr[i_line, :11] = y_cord, x_cord, min, max, std, true_flux, true_err, intg_flux, intg_err, profile_flux, profile_err
    x_arr[i_line, -box_size:] = synth_line[idx_0:idx_f]
    y_arr[i_line] = class_name

    return i_line + 1

# Load sample
cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
params = sample_cfg[f'properties_{version}']
norm = sample_cfg['meta']['scale']
aspect_categories = list(aspect.cfg['number_shape'].values())
include_fit = params['include_fit']

# Categories for the analysis
categories = params['categories']
category_check = {item: item in categories for item in aspect_categories}

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
noise_cont = params['noise_cont']

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
uncrop_array_size = params['uncrop_array_size']
step_arr = np.arange(-uncrop_array_size/2, uncrop_array_size/2, 1)
idx_zero = int(uncrop_array_size/2)
idx_0, idx_f = int(idx_zero - box_pixels/2), int(idx_zero + box_pixels/2)

# --------- Features
cr_boundary = params['cosmic-ray']['cosmic_ray_low_boundary']
white_noise_min_int_ratio = params['white_noise']['min_int_ratio']
white_noise_max_int_ratio = params['white_noise']['max_int_ratio']

abs_max_int_ratio = params['absorption']['max_int_ratio']

doublet_em_res_min = params['doublet_em']['min_res_ratio']
doublet_em_res_max = params['doublet_em']['max_res_ratio']
doublet_em_int_min = params['doublet_em']['min_int_ratio']
doublet_em_int_max = params['doublet_em']['max_int_ratio']
doublet_em_sep_min = params['doublet_em']['min_separation']
doublet_em_sep_max = params['doublet_em']['max_separation']
doublet_em_int_discr = params['doublet_em']['discrepancy_factors']

doublet_abs_res_min = params['doublet_abs']['min_res_ratio']
doublet_abs_res_max = params['doublet_abs']['max_res_ratio']
doublet_abs_int_min = params['doublet_abs']['min_int_ratio']
doublet_abs_int_max = params['doublet_abs']['max_int_ratio']
doublet_abs_sep_min = params['doublet_abs']['min_separation']
doublet_abs_sep_max = params['doublet_abs']['max_separation']
doublet_abs_int_discr = params['doublet_abs']['discrepancy_factors']

# --------- Containers
print('\nGenerating containers for results')
n_lines = sample_size * len(categories)

# Features  Coordinates + Flux measurements +  Scale parameters (min, max, std)  + Scale features    + Pixel Features
n_columns = 2                   + 6                    + 3                                + 1        + box_pixels

pred_arr = np.empty(n_lines, dtype='U20')
data_matrix = np.full((n_lines, n_columns), np.nan)

# ---------  Loop through the conditions
print('\nLooping through combinations')
bar = tqdm(combinations, desc="Item", mininterval=0.2, unit=" combinations")

counter = 0
comps_counter = dict.fromkeys(categories, 0)
for idx, (int_ratio, res_ratio) in enumerate(bar):

    # Continuum components
    wave_arr = mu_line + step_arr * instr_res
    cont_arr = gradient_arr[idx] * wave_arr + cont_level
    noise_i = noise_cont
    white_noise_arr = np.random.normal(loc=0, scale=noise_cont, size=uncrop_array_size)

    # Line components
    amp = int_ratio * noise_i
    sigma = res_ratio * instr_res

    # Reference values
    detection_value = detection_function(res_ratio)
    cosmic_ray_res = cosmic_ray_function(int_ratio, res_ratio_check=False)

    # Detection cases
    if int_ratio >= detection_value:

        # Flux emission array
        flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr

        # Line
        if res_ratio > cosmic_ray_res:
            if category_check['emission']:
                shape = 'emission'
                if comps_counter[shape] < sample_size:
                    local_include = include_fit  # True if (amp >= 3) and (amp <= 50) else include_fit
                    counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels,
                                         amp_i=amp, sigma_i=sigma, noise_arr_i=white_noise_arr, wave_arr_i=wave_arr, include_fit=local_include)
                    comps_counter[shape] += 1

        # Single pixel
        else:
            if category_check['cosmic-ray']:
                local_include = include_fit #True if (amp >= 3) and (amp <= 50) else include_fit
                (amp >= 3) and (amp <= 50)
                if int_ratio > detection_value + cr_boundary:
                    shape = 'cosmic-ray'
                    for i in np.arange(3):
                        if comps_counter[shape] < sample_size:
                            counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels,
                                                 amp_i=amp, sigma_i=sigma, noise_arr_i=white_noise_arr, wave_arr_i=wave_arr, include_fit=local_include)
                            comps_counter[shape] += 1
                else:
                    shape = 'emission'
                    if comps_counter[shape] < sample_size:
                        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels,
                                             amp_i=amp, sigma_i=sigma, noise_arr_i=white_noise_arr, wave_arr_i=wave_arr, include_fit=local_include)
                        comps_counter[shape] += 1

        # Flux absorption array:
        flux_arr = gaussian_model(wave_arr, -amp, mu_line, sigma) + white_noise_arr + cont_arr

        # Line
        if res_ratio > cosmic_ray_res:
            if int_ratio < abs_max_int_ratio:
                if category_check['absorption']:
                    shape = 'absorption'
                    if comps_counter[shape] < sample_size:
                        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                        comps_counter[shape] += 1

        # Dead pixel
        else:
            if category_check['dead-pixel']:
                if int_ratio > detection_value + cr_boundary:
                    shape = 'dead-pixel'
                    for i in np.arange(3):
                        if comps_counter[shape] < sample_size:
                            counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                            comps_counter[shape] += 1
                else:
                    shape = 'absorption'
                    if comps_counter[shape] < sample_size:
                        counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                        comps_counter[shape] += 1

    # Continuum
    else:

        # White noise
        if (int_ratio >= white_noise_min_int_ratio) and (int_ratio <= white_noise_max_int_ratio):
            if category_check['white-noise']:
                shape = 'white-noise'
                if comps_counter[shape] < sample_size:
                    white_noise_arr = np.random.normal(loc=0, scale=1/int_ratio, size=params['uncrop_array_size'])
                    flux_arr = white_noise_arr + cont_arr
                    counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                    comps_counter[shape] += 1

        # Continuum
        else:
            if category_check[ 'continuum']:
                shape = 'continuum'
                if comps_counter[shape] < sample_size:
                    flux_arr = gaussian_model(wave_arr, amp, mu_line, sigma) + white_noise_arr + cont_arr
                    counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                    comps_counter[shape] += 1

    # Doublet-em
    if (int_ratio > doublet_em_int_min) & (int_ratio < doublet_em_int_max) & (res_ratio > doublet_em_res_min) & (res_ratio < doublet_em_res_max):

        if category_check['doublet-em']:

            # Generate the profile
            shape = 'doublet-em'
            if comps_counter[shape] < sample_size:
                flux_arr = doublet_model(wave_arr, white_noise_arr, cont_arr, amp, mu_line, sigma,
                                         doublet_em_sep_min, doublet_em_sep_max,
                                         doublet_em_int_discr[0], doublet_em_int_discr[1],
                                         20, 5000)
                counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                comps_counter[shape] += 1

    # Doublet-abs
    if (int_ratio > doublet_abs_int_min) & (int_ratio < doublet_abs_int_max) & (res_ratio > doublet_abs_res_min) & (res_ratio < doublet_abs_res_max):

        if category_check['doublet-abs']:

            # Generate the profile
            shape = 'doublet-abs'
            for i in np.arange(2):
                if comps_counter[shape] < sample_size:
                    flux_arr = doublet_model(wave_arr, white_noise_arr, cont_arr, -amp, mu_line, sigma,
                                             doublet_em_sep_min, doublet_em_sep_max,
                                             doublet_em_int_discr[0], doublet_em_int_discr[1],
                                             20, 500)
                    counter = store_line(data_matrix, pred_arr, shape, counter, flux_arr, res_ratio, int_ratio, box_pixels)
                    comps_counter[shape] += 1


# ---------  Save the results
idcs_empty = pred_arr != ''
data_matrix = data_matrix[idcs_empty, :]
pred_arr = pred_arr[idcs_empty]

# Crop the dataset to use the same number of points as the smallest number of points
data_matrix, pred_arr, _ = stratify_sample(data_matrix, pred_arr, randomize=True)

# Two individual files per sample
np.savetxt(output_folder/f'data_array_{version}.txt', data_matrix, fmt='%.6f', delimiter=',')
np.savetxt(output_folder/f'pred_array_{version}.txt', pred_arr, fmt='%s')

# Clear the memory just in case
del data_matrix
del pred_arr
gc.collect()

# Load and save a scaled version
data_matrix = np.loadtxt(output_folder/f'data_array_{version}.txt', delimiter=',')
aspect.tools.scale_min_max(data_matrix, box_pixels, axis=1, scale_parameter=norm)
np.savetxt(output_folder/f'data_array_{norm}_{version}.txt', data_matrix, fmt='%.6f', delimiter=',')