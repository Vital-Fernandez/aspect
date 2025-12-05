import matplotlib.pyplot as plt
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
miss_match_objects = [4, 18, 19, 26, 33, 36, 44, 53, 55, 56, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 76,
                      77, 78, 80, 82, 83, 84, 86, 88, 89, 90, 92, 93, 96, 99]
                                    #priority 67, 64, 26 33 44 55, 96
special = [19, 26, 33, 36, 44, 53, 55, 64, 67, 71, 73, 76, 77, 78, 80, 84, 89, 93, 96, 99]
# 84 is very good and 93 is bad (but fitting...)... why?
miss_match_objects = [67]
for i, idx in enumerate(sample_df.index):

    if i in miss_match_objects:

        # Read the spectrum
        z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
        spec_path = spec_dir/root/fname
        print(f'{i}) Object: {fname}, z_true = {z_obj}')

        spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
        spec.unit_conversion('AA', 'FLAM', norm_flux=1e-22)


        # Components detection
        spec.infer.components(exclude_continuum=False)
        # spec.plot.spectrum(show_components=True)
        # spec.plot.spectrum(show_components=True, show_err=True, rest_frame=True)

        bands_obj = spec.retrieve.lines_frame(ref_bands=lines_df)
        spec.plot.spectrum(bands=bands_obj, show_components=True)

        # y_arr = np.diff(lines_df.wavelength.to_numpy())
        # x_arr = np.arange(y_arr.size)
        # print(x_arr)
        # print(lines_df.wavelength.to_numpy())
        #
        # fig, ax = plt.subplots()
        # ax.scatter(x_arr, y_arr)
        # ax.set_xticks(x_arr)
        # ax.set_xticklabels(lines_df.index.to_numpy()[:-1], rotation=60)
        # ax.set_yscale('log')
        # plt.show()

        # Redshift fitting
        z_key = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='key', plot_results=True)
        z_xor = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='xor', plot_results=True)

        # Store if not none
        if z_key is not None: sample_df.loc[idx, 'zkey'] = z_key
        if z_xor is not None: sample_df.loc[idx, 'zxor'] = z_xor

        # if ~np.isclose(z_key, z_xor, rtol=0.05) & ~np.isclose(z_key, z_obj, rtol=0.05):
        #     print(f'{i}) Object: {fname}, z')
        #     print(f' - z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}')

# Save the redshift measurements
# sample_df = lime.save_frame(sample_fname, sample_df)

