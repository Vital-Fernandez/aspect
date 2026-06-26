import lime
from pathlib import Path



fname = './aspect_DAWN_prism_v4_selection.csv'
sample_df = lime.load_frame(fname)

# idcs_prism = sample_df.file.str.contains('prism')
# sample_prism = sample_df.loc[idcs_prism]
# lime.save_frame('./aspect_DAWN_prism_v4_selection.csv', sample_prism)

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()

# Loop through the files and plot them
for idx in sample_df.index:
    z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
    spec_path = spec_dir/root/fname

    spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
    spec.unit_conversion('AA', 'FLAM')

    plot_fname = spec_dir / root / fname.replace('.fits', '.png')
    spec.plot.spectrum(show_err=True, rest_frame=True)

