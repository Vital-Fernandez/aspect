import numpy as np
import lime
from time import time
from pathlib import Path
from aspect.workflow import model_mgr


print(f'Antes: {model_mgr.model_address}')
mname = '/home/vital/Astrodata/aspect/medium_box/results/classifier-v12-RF_min-max-log_12-pixels.joblib'
# mname = '/home/vital/Astrodata/aspect/medium_box/results/classifier-v12-MLP_min-max-log_12-pixels.joblib'
model_mgr.reload_model(model_address=mname, n_jobs=4)
print(f'Despues: {model_mgr.model_address}')

# Read sample database
sample_fname = './aspect_DAWN_prism_v3_selection.csv'
sample_outname = './aspect_DAWN_prism_v3_measurements_randomforest_v4.csv'
sample_df = lime.load_frame(sample_fname)


# Just the prism
idcs = sample_df.file.str.contains('prism')
sample_df = sample_df.loc[idcs]
n_objs = sample_df.index.size

# Read lines file
lines_fname = './redshift_ref_lines.txt'
lines_df = lime.load_frame(lines_fname)

lines_redshift = ['H1_1216A', 'O3_5007A', 'H1_6563A', 'S3_9530A', 'He1_10832A',  'H1_12822A', 'H1_18756A', 'H1_26259A']
lines_df = lines_df.loc[lines_df.index.isin(lines_redshift)]

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()

# Loop throught the objects
start_time = time()
for i, idx in enumerate(sample_df.index):

    if i >= 0:

        # Read the spectrum
        z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
        spec_path = spec_dir/root/fname

        spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
        spec.unit_conversion('AA', 'FLAM', norm_flux=1e-22)

        # Components detection
        spec.infer.components(exclude_continuum=False)

        # Redshift fitting
        z_key = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='key', plot_results=False)
        z_xor = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='xor', plot_results=False)

        # Store if not none
        if z_key is not None: sample_df.loc[idx, 'zkey'] = z_key
        if z_xor is not None: sample_df.loc[idx, 'zxor'] = z_xor

        if (z_key is not None) and (z_xor is not None):
            if ~np.isclose(z_key, z_xor, rtol=0.05) & ~np.isclose(z_key, z_obj, rtol=0.05):
                print(f'{i}/{n_objs}) Object: {fname}, z_true = {z_obj}')
                print(f' - Missmatch: z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}')
        else:
            print(f'{i}/{n_objs}) Object: {fname}, z_true = {z_obj}')
            print(f' - ISSUE: z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}')

# Output
end_time = np.round((time() - start_time) / 60, 2)
print(f'- completed ({end_time} minutes)')

# Save the redshift measurements
sample_df = lime.save_frame(sample_outname, sample_df)
