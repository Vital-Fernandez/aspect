import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib import pyplot as plt


# Function to open nirspec fits files
def load_nirspec_fits(file_address, ext=None):

    # Stablish the file type
    if 'x1d' in file_address:
        ext = 1
        spec_type = 'x1d'

    elif 's2d' in file_address:
        ext = 1
        spec_type = 's2d'

    elif 'uncal' in file_address:
        ext = 1
        spec_type = 's2d'

    elif 'cal' in file_address:
        ext = 1
        spec_type = 'cal'

    else:
        print('Spectrum type could not be guessed')

    # Open the fits file
    with fits.open(file_address) as hdu_list:

        if spec_type == 'x1d':
            data_table, header = hdu_list[ext].data, (hdu_list[0].header, hdu_list[ext].header)
            wave_array, flux_array, err_array = data_table['WAVELENGTH'], data_table['FLUX'], data_table['FLUX_ERROR']

        elif spec_type == 'cal':
            wave_array, flux_array, err_array = None, None, None

        elif spec_type == 's2d':
            header = (hdu_list[0].header, hdu_list[1].header)
            wave_array = np.linspace(header[1]['WAVSTART'], header[1]['WAVEND'], header[1]['NAXIS1'], endpoint=True) * 1000000
            err_array = hdu_list[2].data
            flux_array = hdu_list[1].data

    return wave_array, flux_array, err_array, header

# Function to plot 2D spectra
def plot_2D(wave_spec, flux_spec, idcs=[0, -1], title=None):

    fig, ax = plt.subplots(figsize=(8, 6))

    Z_FUNC_CMAP = ZScaleInterval()

    z1, z2 = Z_FUNC_CMAP.get_limits(flux_spec[:, idcs[0]:idcs[-1]])

    disp_low, disp_high = wave_spec[idcs[0]], wave_spec[idcs[-1]-1]
    spa_low, spa_high = 0, flux_spec.shape[0]

    extend = np.array([disp_low, disp_high, spa_high, spa_low])

    im = ax.imshow(flux_spec, cmap='gist_heat', vmin=z1, vmax=z2, aspect=2, origin='lower', interpolation='none')

    if title is not None:
        ax.set_title(title)

    plt.show()

    return


file_address = '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_s2d.fits'

wave_array, flux_array, err_array, header = load_nirspec_fits(file_address)
wave_array = wave_array * 10000

pred_matrix = np.zeros(flux_array.shape)
conf_matrix = np.zeros(flux_array.shape)

# One spec at a time:
for idx in np.arange(flux_array.shape[0]):

    spec_i = lime.Spec()


# plot_2D(wave_array, flux_array, title='MSA1586')