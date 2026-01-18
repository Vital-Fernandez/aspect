import numpy as np
import lime
from time import time
from pathlib import Path


# Read sample database
sample_fname = './aspect_DAWN_prism_v4_selection.csv'
sample_df = lime.load_frame(sample_fname)

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()

# Prepare reference lines
lines_redshift = ['H1_1216A', 'O2_3726A', 'O2_3729A', 'Ne3_3869A', 'H1_4861A', 'O3_4959A',
                  'O3_5007A', 'H1_6563A',  'S3_9530A', 'He1_10832A',  'H1_12822A', 'H1_18756A']
lines_df = lime.lines_frame(line_list=lines_redshift, vacuum_waves=True)

# Loop through the lines and run aspect
start_time = time()
for i, idx in enumerate(sample_df.index):

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

    print(f'{i}) Object: {fname}')
    if (z_key is not None) & (z_xor is not None):
        if ~np.isclose(z_key, z_xor, rtol=0.05) & ~np.isclose(z_key, z_obj, rtol=0.05):
            print(f' - Missmatch z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}')
    else:
        print(f' - z_true = {z_obj}, None entries!')

end_time = np.round((time() - start_time) / 60, 2)
print(f'- completed ({end_time} minutes)')

# Save the redshift measurements
sample_fname = './aspect_DAWN_prism_v4_measurements.csv'
sample_df = lime.save_frame(sample_fname, sample_df)

# (22.65/(4083 - 502)) * 60 ~ 0.38 seconds per galaxy: reading file, unit conversion, running aspect, 2 x z_key measurement
# (25.6/(4084)) * 60 ~ 0.38 seconds per galaxy: reading file, unit conversion, running aspect, 2 x z_key measurement