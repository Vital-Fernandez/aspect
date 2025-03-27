import numpy as np
from pathlib import Path
import lime
from time import time

# Data folder
data_folder = Path('/home/vital/PycharmProjects/lime/examples/sample_data')

# Configuration file
cfg = lime.load_cfg(data_folder/'long_slit.toml')

# Spectra list
object_dict = {'22431':'nirspec', '001586':'nirspec'}
redshift_dict = {'001586': 4.299, '22431': 9.28}
files_dict = {'001586': '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits',
              '22431': '/home/vital/Astrodata/CAPERS/CAPERS_UDS_V0.1/P1/CAPERS_UDS_P1_s000022431_x1d_optext.fits'}

# Loop through the files and measure the lines
for i, items in enumerate(object_dict.items()):


    obj, inst = items
    file_path = Path(files_dict[obj])
    redshift = redshift_dict[obj]

    # Create the observation object
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    if spec.err_flux is not None:
        if np.all(spec.err_flux > 0):
            start_time = time()
            spec.plot.spectrum()
            spec.infer.components(show_steps=True)
            print(f'- completed ({(time() - start_time):0.3f} seconds)')
            spec.plot.spectrum(show_categories=True, rest_frame=True)

    # # Bands the results
    # bands = spec.retrieve.line_bands(band_vsigma=100)
    # # spec.check.bands(data_folder/'bands'/f'{obj}_{inst}_bands.txt', ref_bands=bands, exclude_continua=False)
    #
    # spec.fit.frame(data_folder/'bands'/f'{obj}_{inst}_bands.txt', cfg, id_conf_prefix=f'{obj}_{inst}')
    # # spec.plot.grid()

