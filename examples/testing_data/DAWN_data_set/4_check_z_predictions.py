import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context
from aspect.workflow import model_mgr

lime.theme.set_style('dark')

# Read sample database
sample_fname = '/home/vital/PycharmProjects/aspect/examples/testing_data/DAWN_data_set/aspect_DAWN_prism_v3_measurements_randomforest_v4.csv'
sample_df = lime.load_frame(sample_fname)

# Just the prism
idcs = sample_df.file.str.contains('prism')
sample_df = sample_df.loc[idcs]

# Preselect

# Whole disagree
# idcs = (sample_df['zkey'].notna() & sample_df['zxor'].notna()) & (abs(sample_df['zkey'] - sample_df['zxor']) / sample_df['zkey'].abs() > 0.05)

# # Both agree on wrong
# idcs = (sample_df['zkey'].notna() & sample_df['zxor'].notna()) & (abs(sample_df['zkey'] - sample_df['zxor']) / sample_df['zkey'].abs() < 0.05)
# idcs = idcs & (abs(sample_df['zkey'] - sample_df['z']) / sample_df['z'].abs() > 0.05)
# sample_df = sample_df.loc[idcs]
# n_objs = sample_df.index.size

# # Jus these files
# fits_files = [
#     "abell2744-ddt-v3_prism-clear_2756_40036.spec.fits",
#     "ceers-v3_prism-clear_1345_5337.spec.fits",
#     "abell2744-ddt-v3_prism-clear_2756_20025.spec.fits",
#     "abell2744-ddt-v3_prism-clear_2756_40094.spec.fits",
#     "borg-0859p4114-v3_prism-clear_1747_1221.spec.fits",
#     "capers-udsp1-v3_prism-clear_6368_15577.spec.fits",
#     "capers-udsp1-v3_prism-clear_6368_19765.spec.fits",
#     "capers-udsp1-v3_prism-clear_6368_24501.spec.fits",
#     "capers-udsp1-v3_prism-clear_6368_24678.spec.fits",
#     "capers-udsp2-v3_prism-clear_6368_13915.spec.fits",
#     "capers-udsp3-v3_prism-clear_6368_3184.spec.fits",
#     "capers-udsp3-v3_prism-clear_6368_3581.spec.fits",
#     "capers-udsp3-v3_prism-clear_6368_4157.spec.fits",
#     "capers-udsp3-v3_prism-clear_6368_4227.spec.fits",
#     "capers-udsp5-v3_prism-clear_6368_35372.spec.fits",
#     "capers-udsp5-v3_prism-clear_6368_35483.spec.fits",
#     "ceers-ddt-v3_prism-clear_2750_1823.spec.fits",          # THIS IS A CRITICAL WHY ZOR FAILS
#     "ceers-v3_prism-clear_1345_10386.spec.fits",
#     "ceers-v3_prism-clear_1345_22912.spec.fits",
#     "ceers-v3_prism-clear_1345_23897.spec.fits",
#     "ceers-v3_prism-clear_1345_30780.spec.fits",
# ]
# idcs = sample_df.file.isin(fits_files)


# Read lines file
lines_fname = './redshift_ref_lines.txt'
lines_df = lime.load_frame(lines_fname)

lines_redshift = ['H1_1216A', 'O3_5007A', 'H1_6563A',  'S3_9530A', 'He1_10832A',  'H1_12822A', 'H1_18756A', 'H1_26259A']
lines_df = lines_df.loc[lines_df.index.isin(lines_redshift)]

# Locate the spectra
spec_dir = Path("/home/vital/Astrodata/DAWN")
root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()


'''
Objects of interest

-Check pixel detection
41/126) gto-wide-uds12-v3_prism-clear_1215_4250.spec.fits z_true = 1.9249, z_key = 0.68, z_xor = 0.69 (5% check = False, 1.5)
53/126) jades-gds02-v3_prism-clear_1286_188720.spec.fits z_true = 2.33, z_key = 1.35, z_xor = 1.35 (5% check = False, 0.0)
55/126) jades-gds06-v3_prism-clear_1286_162894.spec.fits z_true = 2.0198, z_key = 0.53, z_xor = 4.82 (5% check = True, 809.4)
56/126) jades-gds07-v3_prism-clear_1286_200080.spec.fits z_true = 1.9966, z_key = 1.25, z_xor = 4.81 (5% check = True, 284.8)
80/126) rubies-egs62-nod-v3_prism-clear_4233_54336.spec.fits z_true = 2.3017, z_key = 2.28, z_xor = 2.39 (5% check = False, 4.8)

1/126) capers-udsp1-v3_prism-clear_6368_15577.spec.fits z_true = 2.3124, z_key = 3.43, z_xor = 5.4 (5% check = True, 57.4)
4/126) capers-udsp3-v3_prism-clear_6368_4157.spec.fits z_true = 1.9077, z_key = 2.91, z_xor = 4.65 (5% check = True, 59.8)
9/126) ceers-v3_prism-clear_1345_10386.spec.fits z_true = 2.0144, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)
11/126) ceers-v3_prism-clear_1345_2693.spec.fits z_true = 3.6749, z_key = 2.47, z_xor = 2.48 (5% check = False, 0.4)
12/126) ceers-v3_prism-clear_1345_4733.spec.fits z_true = 1.8044, z_key = 2.77, z_xor = 2.81 (5% check = False, 1.4)
13/126) ceers-v3_prism-clear_1345_9939.spec.fits z_true = 2.3022, z_key = 3.43, z_xor = 3.43 (5% check = False, 0.0)
15/126) cosmos-transients-v3_prism-clear_6585_64734.spec.fits z_true = 2.474, z_key = 5.71, z_xor = 5.72 (5% check = False, 0.2)
16/126) gds-barrufet-s156-v3_prism-clear_2198_2187.spec.fits z_true = 2.677, z_key = 3.96, z_xor = 3.96 (5% check = False, 0.0)
17/126) gds-barrufet-s67-v3_prism-clear_2198_7151.spec.fits z_true = 3.4807, z_key = 4.88, z_xor = 4.96 (5% check = False, 1.6)
21/126) gds-egami-ddt-v3_prism-clear_6541_204449.spec.fits z_true = 2.6433, z_key = 1.74, z_xor = 3.89 (5% check = True, 123.6)
23/126) glazebrook-egs-v3_prism-clear_2565_17073.spec.fits z_true = 2.2983, z_key = 3.43, z_xor = 5.37 (5% check = True, 56.6)
34/126) gto-wide-cos02-v3_prism-clear_1214_1317.spec.fits z_true = 2.8228, z_key = 4.02, z_xor = 4.11 (5% check = False, 2.2)
39/126) gto-wide-egs2-v3_prism-clear_1213_3024.spec.fits z_true = 0.8259, z_key = 1.54, z_xor = 1.54 (5% check = False, 0.0)
44/126) jades-gdn09-v3_prism-clear_1181_71914.spec.fits z_true = 2.4593, z_key = 3.66, z_xor = 3.67 (5% check = False, 0.3)
48/126) jades-gdn2-v3_prism-clear_1181_29034.spec.fits z_true = 2.0969, z_key = 3.16, z_xor = 3.16 (5% check = False, 0.0)
51/126) jades-gdn2-v3_prism-clear_1181_3608.spec.fits z_true = 4.084, z_key = 5.7, z_xor = 5.75 (5% check = False, 0.9)
71/126) rubies-egs52-nod-v3_prism-clear_4233_9224.spec.fits z_true = 2.3369, z_key = nan, z_xor = nan (5% check = False, nan)
77/126) rubies-egs53-v3_prism-clear_4233_47517.spec.fits z_true = 1.8019, z_key = 2.77, z_xor = 2.8 (5% check = False, 1.1)

- Both fail
4/126) capers-udsp3-v3_prism-clear_6368_4157.spec.fits z_true = 1.9077, z_key = 2.91, z_xor = 4.65 (5% check = True, 59.8)

-xor fails:
0/126) borg-0955p4528-v3_prism-clear_1747_321.spec.fits z_true = 1.0953, z_key = 1.09, z_xor = 6.94 (5% check = True, 536.7)
1/126) capers-udsp1-v3_prism-clear_6368_15577.spec.fits z_true = 2.3124, z_key = 3.43, z_xor = 5.4 (5% check = True, 57.4)
18/126) gds-deep-v3_prism-clear_1210_12005.spec.fits z_true = 1.8625, z_key = 1.86, z_xor = 4.47 (5% check = True, 140.3)
24/126) glazebrook-egs-v3_prism-clear_2565_17754.spec.fits z_true = 2.3029, z_key = 2.3, z_xor = 5.37 (5% check = True, 133.5)
66/126) jades-gds-wide-v3_prism-clear_1180_12928.spec.fits z_true = 2.1337, z_key = 2.13, z_xor = 5.08 (5% check = True, 138.5)
82/126) rubies-egs62-v3_prism-clear_4233_42944.spec.fits z_true = 2.9476, z_key = 2.94, z_xor = 6.81 (5% check = True, 131.6)
 83/126) rubies-egs62-v3_prism-clear_4233_54336.spec.fits z_true = 2.3013, z_key = 2.29, z_xor = 2.32 (5% check = False, 1.3)


- infrared lines


- zkey fails
8/126) ceers-ddt-v3_prism-clear_2750_5336.spec.fits z_true = 1.5429, z_key = 0.92, z_xor = 1.54 (5% check = True, 67.4)


-Both agree on wrong
2/126) capers-udsp1-v3_prism-clear_6368_16504.spec.fits z_true = 2.7478, z_key = 4.03, z_xor = 4.01 (5% check = False, 0.5)
3/126) capers-udsp1-v3_prism-clear_6368_22752.spec.fits z_true = 3.8524, z_key = 2.69, z_xor = 2.75 (5% check = False, 2.2)
5/126) capers-udsp5-v3_prism-clear_6368_28227.spec.fits z_true = 1.9156, z_key = 2.91, z_xor = 2.95 (5% check = False, 1.4)
6/126) ceers-ddt-v3_prism-clear_2750_1823.spec.fits z_true = 1.9373, z_key = 2.95, z_xor = 2.95 (5% check = False, 0.0)
9/126) ceers-v3_prism-clear_1345_10386.spec.fits z_true = 2.0144, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)
10/126) ceers-v3_prism-clear_1345_22912.spec.fits z_true = 3.1252, z_key = 4.41, z_xor = 4.47 (5% check = False, 1.4)
11/126) ceers-v3_prism-clear_1345_2693.spec.fits z_true = 3.6749, z_key = 2.47, z_xor = 2.48 (5% check = False, 0.4)
13/126) ceers-v3_prism-clear_1345_9939.spec.fits z_true = 2.3022, z_key = 3.43, z_xor = 3.43 (5% check = False, 0.0)
14/126) cosmos-transients-v3_prism-clear_6585_62269.spec.fits z_true = 3.4224, z_key = 4.82, z_xor = 4.88 (5% check = False, 1.2)
15/126) cosmos-transients-v3_prism-clear_6585_64734.spec.fits z_true = 2.474, z_key = 5.71, z_xor = 5.72 (5% check = False, 0.2)
16/126) gds-barrufet-s156-v3_prism-clear_2198_2187.spec.fits z_true = 2.677, z_key = 3.96, z_xor = 3.96 (5% check = False, 0.0)
17/126) gds-barrufet-s67-v3_prism-clear_2198_7151.spec.fits z_true = 3.4807, z_key = 4.88, z_xor = 4.96 (5% check = False, 1.6)
19/126) gds-deep-v3_prism-clear_1210_36834.spec.fits z_true = 2.619, z_key = 3.89, z_xor = 3.89 (5% check = False, 0.0)
20/126) gds-egami-ddt-v3_prism-clear_6541_2022034.spec.fits z_true = 2.6324, z_key = 3.9, z_xor = 3.91 (5% check = False, 0.3)
22/126) glazebrook-cos-obs1-v3_prism-clear_2565_9775.spec.fits z_true = 4.3875, z_key = 6.12, z_xor = 6.14 (5% check = False, 0.3)
42/126) gto-wide-uds12-v3_prism-clear_1215_7007.spec.fits z_true = 2.5967, z_key = 3.83, z_xor = 3.83 (5% check = False, 0.0)
47/126) jades-gdn2-v3_prism-clear_1181_28626.spec.fits z_true = 1.9991, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)

25/126) glazebrook-egs-v3_prism-clear_2565_28777.spec.fits z_true = 3.639, z_key = 5.11, z_xor = 5.16 (5% check = False, 1.0)
26/126) glazebrook-v3_prism-clear_2565_39102.spec.fits z_true = 3.5975, z_key = 5.04, z_xor = 5.11 (5% check = False, 1.4)
30/126) goodsn-wide7-v3_prism-clear_1211_32.spec.fits z_true = 2.9602, z_key = 4.31, z_xor = 4.31 (5% check = False, 0.0)
31/126) goodsn-wide8-v3_prism-clear_1211_4151.spec.fits z_true = 2.233, z_key = 0.82, z_xor = 0.86 (5% check = False, 4.9)
75/126) rubies-egs53-v3_prism-clear_4233_25712.spec.fits z_true = 3.8506, z_key = 5.41, z_xor = 5.44 (5% check = False, 0.6)
76/126) rubies-egs53-v3_prism-clear_4233_36052.spec.fits z_true = 2.8683, z_key = 6.43, z_xor = 6.44 (5% check = False, 0.2)
77/126) rubies-egs53-v3_prism-clear_4233_47517.spec.fits z_true = 1.8019, z_key = 2.77, z_xor = 2.8 (5% check = False, 1.1)
81/126) rubies-egs62-nod-v3_prism-clear_4233_57455.spec.fits z_true = 3.4417, z_key = 4.85, z_xor = 4.9 (5% check = False, 1.0)

# Ingrared lines
31/126) goodsn-wide8-v3_prism-clear_1211_4151.spec.fits z_true = 2.233, z_key = 0.82, z_xor = 0.86 (5% check = False, 4.9)
32/126) gto-wide-cos01-v3_prism-clear_1214_8224.spec.fits z_true = 0.7014, z_key = 6.31, z_xor = 6.31 (5% check = False, 0.0)
36/126) gto-wide-egs1-v3_prism-clear_1213_2859.spec.fits z_true = 0.5571, z_key = 1.17, z_xor = 1.17 (5% check = False, 0.0)
77/126) rubies-egs53-v3_prism-clear_4233_47517.spec.fits z_true = 1.8019, z_key = 2.77, z_xor = 2.8 (5% check = False, 1.1)



# Check 

'''


fits_files = [
    "capers-udsp1-v3_prism-clear_6368_16504.spec.fits",
    "capers-udsp1-v3_prism-clear_6368_22752.spec.fits",
    "capers-udsp5-v3_prism-clear_6368_28227.spec.fits",
    "ceers-ddt-v3_prism-clear_2750_1823.spec.fits",
    "ceers-v3_prism-clear_1345_10386.spec.fits",
    "ceers-v3_prism-clear_1345_22912.spec.fits",
    "ceers-v3_prism-clear_1345_2693.spec.fits",
    "ceers-v3_prism-clear_1345_9939.spec.fits",
    "cosmos-transients-v3_prism-clear_6585_62269.spec.fits",
    "cosmos-transients-v3_prism-clear_6585_64734.spec.fits",
    "gds-barrufet-s156-v3_prism-clear_2198_2187.spec.fits",
    "gds-barrufet-s67-v3_prism-clear_2198_7151.spec.fits",
    "gds-deep-v3_prism-clear_1210_36834.spec.fits",
    "gds-egami-ddt-v3_prism-clear_6541_2022034.spec.fits",
    "glazebrook-cos-obs1-v3_prism-clear_2565_9775.spec.fits",
    "gto-wide-uds12-v3_prism-clear_1215_7007.spec.fits",
    "jades-gdn2-v3_prism-clear_1181_28626.spec.fits",
    "glazebrook-egs-v3_prism-clear_2565_28777.spec.fits",
    "glazebrook-v3_prism-clear_2565_39102.spec.fits",
    "goodsn-wide7-v3_prism-clear_1211_32.spec.fits",
    "goodsn-wide8-v3_prism-clear_1211_4151.spec.fits",
    "rubies-egs53-v3_prism-clear_4233_25712.spec.fits",
    "rubies-egs53-v3_prism-clear_4233_36052.spec.fits",
    "rubies-egs53-v3_prism-clear_4233_47517.spec.fits",
    "rubies-egs62-nod-v3_prism-clear_4233_57455.spec.fits"]

fits_files_3 = [
    "goodsn-wide8-v3_prism-clear_1211_4151.spec.fits",
    "gto-wide-cos01-v3_prism-clear_1214_8224.spec.fits",
    "gto-wide-egs1-v3_prism-clear_1213_2859.spec.fits",
    "rubies-egs53-v3_prism-clear_4233_47517.spec.fits",
]

fits_files_4 = [
    "gto-wide-uds12-v3_prism-clear_1215_4250.spec.fits",
    "jades-gds02-v3_prism-clear_1286_188720.spec.fits",
    "jades-gds06-v3_prism-clear_1286_162894.spec.fits",
    "jades-gds07-v3_prism-clear_1286_200080.spec.fits",
    "rubies-egs62-nod-v3_prism-clear_4233_54336.spec.fits",
    "capers-udsp1-v3_prism-clear_6368_15577.spec.fits",
    "capers-udsp3-v3_prism-clear_6368_4157.spec.fits",
    "ceers-v3_prism-clear_1345_10386.spec.fits",
    "ceers-v3_prism-clear_1345_2693.spec.fits",
    "ceers-v3_prism-clear_1345_4733.spec.fits",
    "ceers-v3_prism-clear_1345_9939.spec.fits",
    "cosmos-transients-v3_prism-clear_6585_64734.spec.fits",
    "gds-barrufet-s156-v3_prism-clear_2198_2187.spec.fits",
    "gds-barrufet-s67-v3_prism-clear_2198_7151.spec.fits",
    "gds-egami-ddt-v3_prism-clear_6541_204449.spec.fits",
    "glazebrook-egs-v3_prism-clear_2565_17073.spec.fits",
    "gto-wide-cos02-v3_prism-clear_1214_1317.spec.fits",
    "gto-wide-egs2-v3_prism-clear_1213_3024.spec.fits",
    "jades-gdn09-v3_prism-clear_1181_71914.spec.fits",
    "jades-gdn2-v3_prism-clear_1181_29034.spec.fits",
    "jades-gdn2-v3_prism-clear_1181_3608.spec.fits",
    "rubies-egs52-nod-v3_prism-clear_4233_9224.spec.fits",
    "rubies-egs53-v3_prism-clear_4233_47517.spec.fits",
]

idcs = sample_df.file.isin(fits_files_4)
sample_df = sample_df.loc[idcs]
n_objs = sample_df.index.size

print(f'Antes: {model_mgr.model_address}')
mname = '/home/vital/Astrodata/aspect/medium_box/results/aspect_min-max-log_12_pixels_v12_randomforest_model.joblib'
mname = '/home/vital/Astrodata/aspect/medium_box/results/aspect_min-max-log_12_pixels_v12_MLP_model.joblib'
# model_mgr.reload_model(model_address=mname, n_jobs=4)
print(f'Despues: {model_mgr.model_address}')


# Loop throught the objects
for i, idx in enumerate(sample_df.index):

    if i >= 16:

        # Read the spectrum
        z_obj, root, fname,  = sample_df.loc[idx, ['z', 'root', 'file']]
        spec_path = spec_dir/root/fname

        spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=z_obj)
        spec.unit_conversion('AA', 'FLAM', norm_flux=1e-22)
        spec.plot.spectrum(show_err=True)

        # Components detection
        spec.infer.components(exclude_continuum=False, plot_steps=True)

        title = f'z_true = {sample_df.loc[idx,"z"]}, z_key = {sample_df.loc[idx,"zkey"]}, z_xor = {sample_df.loc[idx,"zxor"]}'
        print(f'{i}/{n_objs}) {fname} {title}')

        # Redshift fitting
        # obj_bands = spec.retrieve.lines_frame(band_vsigma=140, ref_bands=lines_df)
        # spec.plot.spectrum(bands=obj_bands, rest_frame=True)
        z_key = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='key', plot_results=False)
        z_xor = spec.fit.redshift(lines_df, band_vsigma=140, z_min=0.2, z_max=10, delta_z=0.005, mode='xor', plot_results=False)

        # # Store if not none
        # if z_key is not None: sample_df.loc[idx, 'zkey'] = z_key
        # if z_xor is not None: sample_df.loc[idx, 'zxor'] = z_xor

        diag = (abs(z_key - z_xor) / abs(z_key)) > 0.05
        title = f'z_true = {z_obj}, z_key = {z_key}, z_xor = {z_xor}'
        print(f'{i}/{n_objs}) {fname} {title} (5% check = {diag}, {100*(abs(z_key - z_xor) / abs(z_key)):0.1f})')
        spec.plot.spectrum(rest_frame=False, show_components=True, show_err=True,
                           ax_cfg={'title':title}, fig_cfg={'figure.figsize':(10,5), 'figure.dpi':250})


# # Loop through the files and plot them
# miss_match_objects = [4, 18, 19, 26, 33, 36, 44, 53, 55, 56, 59, 60, 62, 63, 64, 66, 67, 68, 70, 71, 73, 74, 76,
#                       77, 78, 80, 82, 83, 84, 86, 88, 89, 90, 92, 93, 96, 99]
#                                     #priority 67, 64, 26 33 44 55, 96
# special = [19, 26, 33, 36, 44, 53, 55, 64, 67, 71, 73, 76, 77, 78, 80, 84, 89, 93, 96, 99]
# # 84 is very good and 93 is bad (but fitting...)... why?
#
# miss_match_objects = [67]


'''
Objects of interest

Check pixel detection
2/6459) abell2744-castellano1-v3_prism-clear_3073_16874.spec.fits z_true = 3.4758, z_key = nan, z_xor = nan
15/367) capers-udsp5-v3_prism-clear_6368_28227.spec.fits z_true = 1.9156, z_key = 2.91, z_xor = 2.95 (5% check = False, 1.4)
21/367) ceers-v3_prism-clear_1345_10386.spec.fits z_true = 2.0144, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)
27/367) ceers-v3_prism-clear_1345_2781.spec.fits z_true = 2.2417, z_key = nan, z_xor = nan (5% check = False, nan)
28/367) ceers-v3_prism-clear_1345_30780.spec.fits z_true = 1.6467, z_key = 2.48, z_xor = 2.53 (5% check = False, 2.0)
16/21) ceers-v3_prism-clear_1345_10386.spec.fits z_true = 2.0144, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)
19/21) ceers-v3_prism-clear_1345_30780.spec.fits z_true = 1.6467, z_key = 1.64, z_xor = 1.63 (5% check = False, 0.6)

Why wrong redshift? (key method problem)
4/6459) abell2744-castellano1-v3_prism-clear_3073_17437.spec.fits z_true = 5.5987, z_key = 5.59, z_xor = 7.69
5/6459) abell2744-castellano1-v3_prism-clear_3073_18678.spec.fits z_true = 0.6041, z_key = 0.61, z_xor = 1.17
6/6459) abell2744-castellano1-v3_prism-clear_3073_20605.spec.fits z_true = 5.6438, z_key = 5.63, z_xor = 7.72
8/6459) abell2744-castellano1-v3_prism-clear_3073_21148.spec.fits z_true = 3.4767, z_key = 3.48, z_xor = 4.91

xor fails:
17/6459) abell2744-ddt-v3_prism-clear_2756_110003.spec.fits z_true = 5.6612, z_key = 5.66, z_xor = 7.76 (5% check = True, 37.1)
11/6459) abell2744-castellano1-v3_prism-clear_3073_22901.spec.fits z_true = 3.9658, z_key = 3.96, z_xor = 5.59 (5% check = True, 41.2)
12/6459) abell2744-castellano1-v3_prism-clear_3073_23401.spec.fits z_true = 2.8201, z_key = 2.8, z_xor = 4.01 (5% check = True, 43.2)
13/6459) abell2744-castellano1-v3_prism-clear_3073_23622.spec.fits z_true = 5.23, z_key = 5.23, z_xor = 7.2 (5% check = True, 37.7)
14/6459) abell2744-castellano1-v3_prism-clear_3073_23879.spec.fits z_true = 4.5095, z_key = 4.5, z_xor = 6.3 (5% check = True, 40.0)
15/6459) abell2744-castellano1-v3_prism-clear_3073_24194.spec.fits z_true = 3.3231, z_key = 3.31, z_xor = 4.72 (5% check = True, 42.6)
17/6459) abell2744-ddt-v3_prism-clear_2756_110003.spec.fits z_true = 5.6612, z_key = 5.66, z_xor = 7.76 (5% check = True, 37.1)
18/6459) abell2744-ddt-v3_prism-clear_2756_150015.spec.fits z_true = 5.0479, z_key = 5.04, z_xor = 6.96 (5% check = True, 38.1)
19/6459) abell2744-ddt-v3_prism-clear_2756_160133.spec.fits z_true = 4.0224, z_key = 4.02, z_xor = 5.68 (5% check = True, 41.3)
lyman alpha
20/6459) abell2744-ddt-v3_prism-clear_2756_160170.spec.fits z_true = 4.9109, z_key = 4.89, z_xor = 4.96 (5% check = False, 1.4)
21/6459) abell2744-ddt-v3_prism-clear_2756_160185.spec.fits z_true = 3.4696, z_key = 3.46, z_xor = 4.91 (5% check = True, 41.9)
24/6459) abell2744-ddt-v3_prism-clear_2756_301.spec.fits z_true = 3.9859, z_key = 3.98, z_xor = 5.59 (5% check = True, 40.5)
26/6459) abell2744-ddt-v3_prism-clear_2756_40020.spec.fits z_true = 4.7265, z_key = 4.72, z_xor = 6.59 (5% check = True, 39.6)

infrared lines
28/6459) abell2744-ddt-v3_prism-clear_2756_40066.spec.fits z_true = 4.0229, z_key = 4.02, z_xor = 5.68 (5% check = True, 41.3)
19/367) ceers-ddt-v3_prism-clear_2750_1823.spec.fits z_true = 1.9373, z_key = 2.95, z_xor = 2.95 (5% check = False, 0.0)

Both disagree not on correct
16/6459) abell2744-castellano1-v3_prism-clear_3073_24245.spec.fits z_true = 1.2705, z_key = 2.93, z_xor = 8.86 (5% check = True, 202.4)
22/6459) abell2744-ddt-v3_prism-clear_2756_160284.spec.fits z_true = 4.7042, z_key = 0.52, z_xor = 6.56 (5% check = True, 1161.5)

zkey fails
23/6459) abell2744-ddt-v3_prism-clear_2756_20021.spec.fits z_true = 1.3698, z_key = 2.15, z_xor = 1.31 (5% check = True, 39.1)


Both agree on wrong
27/6459) abell2744-ddt-v3_prism-clear_2756_40036.spec.fits z_true = 3.0613, z_key = 4.35, z_xor = 4.4 (5% check = False, 1.1)
35/367) ceers-v3_prism-clear_1345_5337.spec.fits z_true = 2.1421, z_key = 3.19, z_xor = 3.24 (5% check = False, 1.6)
0/367) abell2744-ddt-v3_prism-clear_2756_20025.spec.fits z_true = 1.8602, z_key = 2.78, z_xor = 2.78 (5% check = False, 0.0)
1/367) abell2744-ddt-v3_prism-clear_2756_40094.spec.fits z_true = 2.6697, z_key = 3.86, z_xor = 3.86 (5% check = False, 0.0)
2/367) borg-0859p4114-v3_prism-clear_1747_1221.spec.fits z_true = 2.6995, z_key = 3.86, z_xor = 3.96 (5% check = False, 2.6)
3/367) capers-udsp1-v3_prism-clear_6368_15577.spec.fits z_true = 2.3124, z_key = 3.43, z_xor = 5.4 (5% check = True, 57.4)
4/367) capers-udsp1-v3_prism-clear_6368_19765.spec.fits z_true = 2.7426, z_key = 3.96, z_xor = 4.01 (5% check = False, 1.3)
6/367) capers-udsp1-v3_prism-clear_6368_24501.spec.fits z_true = 1.2791, z_key = 1.29, z_xor = 1.29 (5% check = False, 0.0)
7/367) capers-udsp1-v3_prism-clear_6368_24678.spec.fits z_true = 2.3032, z_key = 3.4, z_xor = 3.4 (5% check = False, 0.0)
9/367) capers-udsp2-v3_prism-clear_6368_13915.spec.fits z_true = 3.2338, z_key = 4.68, z_xor = 4.63 (5% check = False, 1.1)
10/367) capers-udsp3-v3_prism-clear_6368_3184.spec.fits z_true = 1.2264, z_key = 1.93, z_xor = 1.85 (5% check = False, 4.1)
11/367) capers-udsp3-v3_prism-clear_6368_3581.spec.fits z_true = 3.1907, z_key = 4.63, z_xor = 4.59 (5% check = False, 0.9)
12/367) capers-udsp3-v3_prism-clear_6368_4157.spec.fits z_true = 1.9077, z_key = 2.91, z_xor = 2.91 (5% check = False, 0.0)
13/367) capers-udsp3-v3_prism-clear_6368_4227.spec.fits z_true = 2.0845, z_key = 3.09, z_xor = 3.09 (5% check = False, 0.0)
16/367) capers-udsp5-v3_prism-clear_6368_35372.spec.fits z_true = 3.2579, z_key = 4.72, z_xor = 4.68 (5% check = False, 0.8)
17/367) capers-udsp5-v3_prism-clear_6368_35483.spec.fits z_true = 2.3167, z_key = 3.4, z_xor = 3.5 (5% check = False, 2.9)
19/367) ceers-ddt-v3_prism-clear_2750_1823.spec.fits z_true = 1.9373, z_key = 2.95, z_xor = 2.95 (5% check = False, 0.0)
21/367) ceers-v3_prism-clear_1345_10386.spec.fits z_true = 2.0144, z_key = 3.04, z_xor = 3.09 (5% check = False, 1.6)
25/367) ceers-v3_prism-clear_1345_22912.spec.fits z_true = 3.1252, z_key = 4.41, z_xor = 4.47 (5% check = False, 1.4)
26/367) ceers-v3_prism-clear_1345_23897.spec.fits z_true = 2.374, z_key = 3.5, z_xor = 3.5 (5% check = False, 0.0)
28/367) ceers-v3_prism-clear_1345_30780.spec.fits z_true = 1.6467, z_key = 2.48, z_xor = 2.53 (5% check = False, 2.0)

# Good example witha  mask
5/367) capers-udsp1-v3_prism-clear_6368_22235.spec.fits z_true = 1.6163, z_key = 2.48, z_xor = 2.53 (5% check = False, 2.0)

# Good example foreground object
1/24) gto-wide-uds12-v3_prism-clear_1215_4250.spec.fits z_true = 1.9249, z_key = 1.92, z_xor = 0.7 (5% check = True, 63.5)


# Check 
'''