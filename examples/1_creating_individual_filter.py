# import lime
#
# file = '/home/vital/Downloads/spec-0376-52143-0160.fits'
# bands_file = 'PHL293_bands.txt'
# redshift = 0.00527273
#
# spec = lime.Spectrum.from_file(file, instrument='sdss', redshift=redshift)
#
# # bands = spec.retrieve.line_bands(line_list=['H1_6563A'], band_vsigma=350)
# # spec.check.bands(bands_file, ref_bands=bands)
#
# # spec.plot.spectrum(bands=bands_file, rest_frame=True)
# spec.fit.bands('H1_6563A_b', bands_file, 'fit_cfg.toml', id_conf_prefix='PHL293')
# spec.fit.report()
# spec.save_frame('Halpha_measurements.txt')
# spec.plot.bands(rest_frame=True)
#
# # spec.plot.bands('H1_6563A', ref_bands=bands)

import numpy as np
from matplotlib import pyplot as plt

box_size = 12
mu_lambda = 4935.461
instr_resolution = 1
step_arr = np.arange(-box_size/2, box_size/2, 1)
wave_arr = mu_lambda + step_arr * instr_resolution
print(wave_arr)
print(wave_arr[int(box_size/2)] == mu_lambda)
min_wave = mu_lambda - (box_size/2 * instr_resolution)
max_wave = mu_lambda + ((box_size-1)/2 * instr_resolution)
print(min_wave, wave_arr[0])
print(max_wave, wave_arr[-1])

# fig, ax = plt.subplots()
# x = np.array([6, 12, 24, 48])
# res_power = np.array([100, 350])
# instr_resolution = mu_lambda / res_power
# for i, res_value in enumerate(res_power):
#     y = 2 * x/2 * instr_resolution[i]
#     print(instr_resolution[i])
#     ax.plot(x, y, label = r'$R = \frac{\lambda}{\Delta\lambda}$'+ f' = {res_value}')
#
# ax.axhspan(0, 145, alpha=0.5)
#
# ax.update({'title': r'$H\beta$ and [OIII] doublet inteval covered by detection box',
#            'xlabel': r'Box size (pixels)',
#            'ylabel': r'Wave interval ($\AA$)'})
# ax.legend()
# plt.show()ç

fig, ax = plt.subplots()
x = np.array([50, 100, 150, 200, 250])
res_power = np.array([100, 350])
instr_resolution = mu_lambda / res_power
for i, res_value in enumerate(res_power):
    y = x / instr_resolution[i]
    print(instr_resolution[i])
    ax.plot(x, y, label = r'$R = \frac{\lambda}{\Delta\lambda}$'+ f' = {res_value}')

ax.axvline(150, label=r'Width $H\beta$ and [OIII] doublet band')

ax.update({'xlabel': r'Spectral band width (angstroms)',
           'ylabel': r'Box size (pixels)'})
ax.legend()
plt.show()