from pathlib import Path
import lime
import aspect

# Data folder
data_folder = Path('/home/vital/PycharmProjects/lime/examples/sample_data')

# Configuration file
cfg = lime.load_cfg(data_folder/'long_slit.toml')

# Spectra list
object_dict = {'osiris':'gp121903', 'nirspec':'ceers1027', 'isis':'Izw18', 'sdss':'SHOC579'}

# File list
files_dict = {'osiris': 'gp121903_osiris.fits',
              'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'isis': 'IZW18_isis.fits',
              'sdss':'SHOC579_SDSS_dr18.fits'}

# # Spectra list
# object_dict = {'22431':'nirspec', '001586':'nirspec'}
# redshift_dict = {'001586': 4.299, '22431': 9.28}
# files_dict = {'001586': '/home/vital/PycharmProjects/ceers-data/data/spectra/CEERs_DR0.9/nirspecDDT/prism/hlsp_ceers_jwst_nirspec_nirspecDDT-001586_prism_dr0.9_x1d.fits',
#               '22431': '/home/vital/Astrodata/CAPERS/CAPERS_UDS_V0.1/P1/CAPERS_UDS_P1_s000022431_x1d_optext.fits'}

aspect.model_mgr.medium.predictor.n_jobs = 4

# Loop through the files and measure the lines
for i, items in enumerate(object_dict.items()):

    inst, obj = items
    file_path = data_folder/'spectra'/files_dict[inst]
    redshift = cfg[inst][obj]['z']

    # Create the observation object
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift)

    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    # if spec.err_flux is not None:
    #     if np.all(spec.err_flux > 0):
    #         start_time = time()
    #         spec.infer.components(show_steps=True)
    #         print(f'- completed ({(time() - start_time):0.3f} seconds)')

    spec.plot.spectrum(show_components=True, rest_frame=True)
