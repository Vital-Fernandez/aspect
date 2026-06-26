import numpy as np
import lime
from time import time
from pathlib import Path
import aspect
from aspect.workflow import model_mgr

# Read sample database
sample_fname = './aspect_DAWN_prism_v5_flags1-2-3_selection.csv'
sample_df = lime.load_frame(sample_fname)
if "capers_flag" not in sample_df.columns:
    sample_df.insert(sample_df.columns.get_loc("zxor"), "capers_flag", 0)

if "failed_opening" not in sample_df.columns:
    sample_df.insert(sample_df.columns.get_loc("zxor"), "failed_opening", False)


# Read lines file
lines_fname = './redshift_ref_lines.txt'
lines_df = lime.load_frame(lines_fname)

lines_redshift = ['H1_1216A', 'H1_4340A', 'O3_5007A', 'H1_6563A',  'S3_9530A', 'He1_10832A',  'H1_12822A', 'H1_18756A', 'H1_26259A']
lines_df = lines_df.loc[lines_df.index.isin(lines_redshift)]

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()

# Loop through the lines and run aspect
start_time = time()
for i, idx in enumerate(sample_df.index):

        # Read the spectrum
        z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
        spec_path = spec_dir/root/fname

        try:
            spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
            spec.unit_conversion('AA', 'FLAM', norm_flux=1e-22)
            valid = True
        except:
            valid = False
            sample_df.loc[idx, 'failed_opening'] = True

        if valid:

            # Components detection
            spec.infer.components(exclude_continuum=False)

            # Redshift fitting
            z_key = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='key')
            z_xor = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='xor')
            capers_flag = None

            # Flagging the data
            if z_key is not None and z_xor is not None:
                if np.isnan(z_key) and np.isnan(z_xor):
                    capers_flag = 9

            # Store if not none
            if z_key is not None: sample_df.loc[idx, 'zkey'] = z_key
            if z_xor is not None: sample_df.loc[idx, 'zxor'] = z_xor

            print(f'{i}) Object: {fname}')
            if (z_key is not None) & (z_xor is not None):
                if ~np.isclose(z_key, z_xor, rtol=0.05) & ~np.isclose(z_key, z_obj, rtol=0.05):
                    print(f' - Missmatch z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}')
            else:
                print(f' - z_true = {z_obj}, None entries!')

            if capers_flag is not None:
                sample_df.loc[idx, 'capers_flag'] = 9

end_time = np.round((time() - start_time) / 60, 2)
print(f'- completed ({end_time} minutes)')

# Save the redshift measurements
sample_fname = './aspect_DAWN_prism_classifier_v12_RF_v9_flags1-2-3_selection.csv'
sample_df = lime.save_frame(sample_fname, sample_df)