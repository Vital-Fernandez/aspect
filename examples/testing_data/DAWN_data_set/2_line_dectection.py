import numpy as np
import lime
from pathlib import Path


# Read sample database
sample_fname = './aspect_DAWN_prism_v3_selection.csv'
sample_df = lime.load_frame(sample_fname)

# Read lines file
lines_fname = './redshift_ref_lines.txt'
lines_df = lime.load_frame(lines_fname)

lines_redshift = ['H1_1216A', 'O2_3726A', 'O2_3729A', 'Ne3_3869A', 'H1_4861A', 'O3_4959A',
                  'O3_5007A', 'H1_6563A',  'S3_9530A', 'He1_10832A',  'H1_12822A', 'H1_18756A']
lines_df = lines_df.loc[lines_df.index.isin(lines_redshift)]

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()

# Loop through the files and plot them
for i, idx in enumerate(sample_df.index):

    # Read the spectrum
    z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
    spec_path = spec_dir/root/fname
    print(f'{i}) Object: {fname}')

    spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
    spec.unit_conversion('AA', 'FLAM')

    # Components detection
    spec.infer.components(show_steps=False, exclude_continuum=False)
    # spec.plot.spectrum(show_components=True, show_err=True, rest_frame=True)

    # Redshift fitting
    z_key = spec.fit.redshift(lines_df,  sigma_factor=1, z_min=0.2, z_max=10, mode='key', plot_results=False)
    z_xor = spec.fit.redshift(lines_df, sigma_factor=1, z_min=0.2, z_max=10, mode='xor', plot_results=False)

    # Store if not none
    if z_key is not None: sample_df.loc[idx, 'zkey'] = z_key
    if z_xor is not None: sample_df.loc[idx, 'zxor'] = z_xor

# Save the redshift measurements
sample_df = lime.save_frame(sample_fname, sample_df)

