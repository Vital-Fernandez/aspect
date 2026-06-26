import lime
from pathlib import Path
import lime
from aspect.workflow import model_mgr

# State the data files
data_folder = Path('/home/vital/PycharmProjects/lime/examples/doc_notebooks/0_resources/')
cfgFile = f'{data_folder}/long_slit.toml'
osiris_gp_df_path =  f'{data_folder}/bands/osiris_green_peas_linesDF.txt'

# model_fname = '/home/vital/Astrodata/aspect/medium_box/results/aspect_min-max_12_pixels_v11_MLP_flux_model.joblib'
# model_mgr.reload_model(model_address=model_fname, n_jobs=4)
model_mgr.reload_model(model_key='classifier_v12_MLP', n_jobs=4)
print(f'Despues: {model_mgr.model_address}')

# Load configuration
obs_cfg = lime.load_cfg(cfgFile)

# Instrument - file dictionary
files_dict = {'nirspec':'hlsp_ceers_jwst_nirspec_nirspec10-001027_comb-mgrat_v0.7_x1d-masked.fits',
              'sdss':'SHOC579_SDSS_dr18.fits'}

# Instrument - object dictionary
object_dict = {'nirspec':'ceers1027', 'sdss':'SHOC579',}
object_dict = {'sdss':'SHOC579',}

# Loop through the observations
for i, items in enumerate(object_dict.items()):

    inst, obj = items
    file_path = f'{data_folder}/spectra/{files_dict[inst]}'
    redshift = obs_cfg[inst][obj]['z']
    print('\n', obj, inst, redshift)

    # Create the observation object
    spec = lime.Spectrum.from_file(file_path, inst, redshift=redshift, crop_waves=(3860, 45000))

    # Unit conversion for NIRSPEC object
    if spec.units_wave != 'AA':
        spec.unit_conversion('AA', 'FLAM')

    # Detect the components
    print(spec.infer.model_mgr)

    spec.infer.components(exclude_continuum=False, plot_steps=False)
    print(spec.infer.model_mgr)

    # Show the components
    spec.plot.spectrum(show_components=True, rest_frame=False, show_err=True)
