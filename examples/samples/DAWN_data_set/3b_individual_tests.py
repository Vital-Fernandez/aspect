from time import time
from pathlib import Path

import warnings
import numpy as np
from matplotlib import pyplot as plt, rc_context
import lime


'''

ASPECT paper blind redshift measurement script for the the DAWN v3 sample

This should install all the dependencies
pip install lime-stable[full]

The algorithm inputs (the line list and bands width) is optimize for prism measurements. You can test the algorithm settings at
https://specsy.streamlit.app/a_Components_detection

The code is taking about 0.1 seconds per object but the measurements are only saved at the end of the script.

The first two functions are for making the quality flag and the diagnostics plot (These should be part of ASPECT package in the future)

The code is using 'nirspec_grizli' method to open the fits files. For Pablo's reductions use instrument='nirspec'

The algorithm has two techniques to fit the redshift which results in a quality flagging scheme:

0 = No lines detected, 1 = only one line detected, 2 = redshift measurement with disagreement (z_aspect_flux_sum is favored)
3 = redshift measurement with agreement, -1 = Unexpected issue with the redshift measurement.

In the DAWN archive control sample the grading also has 4 categoreis
 (3 = Robust 2 = Perhaps line or continuum features, but ambiguous redshift
1 = No features 0 = DQ problem) 

'''


def flag_measurement(zkey_flux, zkey_pixel, tolerance=0.05):

    """Flag the consistency between two redshift measurements.

    The two inputs (``zkey_flux`` and ``zkey_pixel``) are expected to agree
    for a high-quality measurement. This function compares their states and
    returns an integer quality flag.

    Parameters
    ----------
    zkey_flux, zkey_pixel : float or None
        The two redshift measurements to compare. Each is expected to be a
        float, ``np.nan``, or ``None``.
    tolerance : float, optional
        Maximum relative difference for two numeric measurements to count as
        matching, passed as ``rtol`` to ``np.isclose`` (default 0.05, i.e. 5%).

    Returns
    -------
    int
        Quality flag encoding the comparison:

        - ``0`` : both inputs are ``None``.
        - ``1`` : both inputs are ``np.nan``.
        - ``2`` : both inputs are numeric but differ by more than ``tolerance``.
        - ``3`` : both inputs are numeric and agree within ``tolerance``.
        - ``-1`` : any other combination (a warning is issued).

    Notes
    -----
    ``None`` is checked with ``is None`` before calling ``np.isnan``, so a
    measurement of exactly ``0.0`` is treated as a legitimate number rather
    than a missing value. ``np.isclose`` is called with ``atol=0.0`` so that
    two zero measurements still compare as matching (flag ``3``).
    """
    flux_none = zkey_flux is None
    pixel_none = zkey_pixel is None
    if flux_none and pixel_none:
        return 0

    # Only safe to call np.isnan once None is ruled out.
    try:
        flux_nan = (not flux_none) and np.isnan(zkey_flux)
        pixel_nan = (not pixel_none) and np.isnan(zkey_pixel)
    except (TypeError, ValueError):
        warnings.warn(
            f"Non-numeric input for quality flag: "
            f"zkey_flux={zkey_flux!r}, zkey_pixel={zkey_pixel!r}; returning -1.",
            stacklevel=2,
        )
        return -1

    if flux_nan and pixel_nan:
        return 1

    flux_num = (not flux_none) and (not flux_nan)
    pixel_num = (not pixel_none) and (not pixel_nan)
    if flux_num and pixel_num:
        return 3 if np.isclose(zkey_flux, zkey_pixel, rtol=tolerance, atol=0.0) else 2

    warnings.warn(
        f"Inconsistent inputs for quality flag: "
        f"zkey_flux={zkey_flux!r}, zkey_pixel={zkey_pixel!r}; returning -1.",
        stacklevel=2,
    )
    return -1



def plot_redshift_diagnostic(sample_df, output_address=None, z_true_hdr='z',
                             flag_hdr='z_aspect_flag', idcs_good=None,
                             z_pixel_hdr='z_aspect_pixel_count',
                             z_flux_hdr='z_aspect_flux_sum',
                             z_limit=12, conf_plot=None):

    """Blind redshift comparison diagnostic (scatter + false-negative marginal).

    Parameters
    ----------
    sample_df : pandas.DataFrame
        Table with the true redshift, the two aspect predictions and the flag.
    output_address : str or Path, optional
        Where to save the figure. If ``None`` (default) the figure is shown
        interactively instead of saved.
    z_true_hdr : str, optional
        Column name of the true (reference) redshift (default ``'z'``).
    flag_hdr : str, optional
        Column name of the quality flag (default ``'z_aspect_flag'``).
    idcs_good : array-like of bool, optional
        Boolean mask selecting the good objects. Bad objects are taken as its
        complement (``~idcs_good``). If not given, every object is treated as
        good (and none as bad).
    z_pixel_hdr, z_flux_hdr : str, optional
        Column names of the two aspect predictions (pixel-count and flux-sum).
        Left at the defaults used throughout this project.
    z_limit : float, optional
        Upper axis/bin limit for both redshift axes (default 12).
    conf_plot : dict, optional
        rcParams overrides passed to ``lime.theme.fig_defaults``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """

    n_rows = sample_df.index.size

    # Optional indexes to establish the high | low quality redshift measurements in the true sample
    idcs_good = np.ones(n_rows, dtype=bool) if idcs_good is None else np.asarray(idcs_good, dtype=bool)
    idcs_bad = ~idcs_good

    if conf_plot is None:
        conf_plot = {"figure.dpi": 300, "figure.figsize": [14, 6],
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 10, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "font.size": 5}

    x_label_list = ['pixel count', 'flux sum']
    z_column = [z_pixel_hdr, z_flux_hdr]
    z_other = [z_flux_hdr, z_pixel_hdr]

    flag_col = sample_df[flag_hdr]
    flag_3 = (flag_col == 3).to_numpy()
    flag_2 = (flag_col == 2).to_numpy()
    flag_1 = (flag_col == 1).to_numpy()
    flag_0 = (flag_col == 0).to_numpy()

    # Bins missed measurements
    z_bins = np.arange(0, z_limit + 0.5, 0.5)

    def pct(n, d):  # guard against an empty good/bad selection
        return 0 if d == 0 else np.round(n / d * 100).astype(int)

    n_good, n_bad = idcs_good.sum(), idcs_bad.sum()

    with rc_context(lime.theme.fig_defaults(conf_plot)):

        # Two scatter panels, each with a right-hand (histy) marginal
        mosaic = [['scat_0', 'histy_0', 'scat_1', 'histy_1']]
        fig, axs = plt.subplot_mosaic(mosaic, width_ratios=[4, 1, 4, 1])

        for i, z_param in enumerate(z_column):

            ax = axs[f'scat_{i}']
            ax_histy = axs[f'histy_{i}']
            ax_histy.sharey(ax)

            # Trendline
            ax.plot([0, z_limit], [0, z_limit], '-', color='yellow', linewidth=0.5)

            # Data
            x, y, w = (sample_df.loc[:, z_param].to_numpy(),
                       sample_df.loc[:, z_true_hdr].to_numpy(),
                       sample_df.loc[:, z_other[i]].to_numpy())
            idcs_TP = np.isclose(x, y, rtol=0.05, equal_nan=False)

            # # 1 True positive
            # frac = pct(idcs_TP.sum(), n_good)
            # ax.scatter(x[idcs_TP], y[idcs_TP], alpha=0.5,
            #            label=f'True positive: {frac}% of {n_good}',
            #            color='tab:blue', edgecolors='none')

            # 2 False positive (single technique)
            idcs_FP2 = (flag_2) & ~np.isclose(x, y, rtol=0.05, equal_nan=False)
            frac = pct(idcs_FP2.sum(), n_good)
            ax.scatter(x[idcs_FP2], y[idcs_FP2], alpha=0.5,
                       label=f'False positive: {frac}% of {n_good}',
                       color='tab:orange', edgecolors='none')

            # 3 False positive (both techniques)
            idcs_FP3 = flag_3 & ~np.isclose(x, y, rtol=0.05, equal_nan=False)
            frac = pct(idcs_FP3.sum(), n_good)
            ax.scatter(x[idcs_FP3], y[idcs_FP3], alpha=0.5,
                       label=f'False positive both techniques: {frac}% of {n_good}',
                       color='tab:red', edgecolors='none')

            # # 4 True negatives
            # idcs_TN = idcs_bad & ~(flag_2 | flag_3)
            # frac = pct(idcs_TN.sum(), n_bad)
            # ax.scatter([], [], alpha=0.8, label=f'True negative: {frac}% of {n_bad}',
            #            color='tab:grey', edgecolors='none', marker='s')
            #
            # # 5 False negatives -> distribution of their true redshifts on the right axis
            # idcs_missing = flag_0 if i == 0 else flag_1
            # idcs_FN = idcs_good & idcs_missing
            # z_arr_missing = sample_df.loc[idcs_FN, z_true_hdr].to_numpy()
            # frac = pct(idcs_FN.sum(), n_good)
            # fn_color = 'tab:green' if i == 0 else 'tab:purple'
            #
            # ax_histy.hist(z_arr_missing, bins=z_bins, orientation='horizontal',
            #               color=fn_color, alpha=0.7)
            #
            # # Legend proxy so the category still shows in the scatter legend
            # ax.scatter([], [], alpha=0.7, marker='s', edgecolors='none', color=fn_color,
            #            label=f'No measurement ({"No lines" if i == 0 else "1 line"}): '
            #                  f'{frac}% of {n_good}')

            # Scatter plot formatting
            ax.grid(True, which='both', linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, z_limit)
            ax.set_ylim(0, z_limit)
            ax.set_xlabel(r'$z_{Aspect}$' + f' {x_label_list[i]}')
            if i == 0:
                ax.set_ylabel(r'$z_{true}$')
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.legend(loc='upper left', framealpha=1)

            # Histogram formatting
            ax_histy.tick_params(axis='y', labelleft=False)

        fig.suptitle(f"DAWN archive blind redshift comparison: {n_rows} spectra", fontsize=16)
        plt.tight_layout()

        if output_address is None:
            plt.show()
        else:
            plt.savefig(output_address)

    return fig


# Declare inputs
sample_fname = './aspect_DAWN_prism_control_classifier-v12-RF_min-max-log_12-pixels_sample_v21.csv' # Table with file selection
spec_root = Path("/home/vital/Astrodata/DAWN_1") # Root folder for spectra

# Add columns to store measurements
sample_df = lime.load_frame(sample_fname)


# sample_df['z_aspect_flux_sum'] = np.nan
# sample_df['z_aspect_pixel_count'] = np.nan
# sample_df['z_aspect_flag'] = np.nan
# sample_df['failed_opening'] = False

# Data
# mask2 = (sample_df['z_aspect_flag'] == 2) & (~np.isclose(sample_df['z_aspect_flux_sum'], sample_df['z'], rtol=0.05, equal_nan=False)) & (sample_df['z'] > 4) & (sample_df['z_aspect_pixel_count'] > 6)
mask2 = (sample_df['z_aspect_flag'] == 2) & (sample_df['z'] > 4) & (sample_df['z_aspect_pixel_count'] > 6)
sample_df = sample_df.loc[mask2]

# Configuration inputs
ref_lines = ['H1_1216A', 'H1_4340A', 'H1_4861A', 'O3_4959A', 'O3_5007A', 'H1_6563A', 'S3_9530A',
             'He1_10832A',  'H1_12822A', 'H1_18756A', 'H1_26259A']
lines_df = lime.lines_frame(line_list=ref_lines)
map_min_R = {250: ['H1_4861A', 'O3_4959A']}
bands_vsigma = 140
z_min, z_max, z_step = 0.1, 15, 0.005


# Loop through the spectra and run aspect
run_fit = True
if run_fit:
    start_time = time()
    root_arr, file_arr = sample_df.root.to_numpy(), sample_df.file.to_numpy()
    for i, idx in enumerate(sample_df.index):

        if i >= 0:

            # Read the spectrum
            folder, fname = sample_df.loc[idx, ['root', 'file']]
            spec_path = spec_root/folder/fname
            print(f'{i}/{len(sample_df.index)}) {fname}')

            # In case file is corrupted
            try:
                spec = lime.Spectrum.from_file(spec_path, instrument='nirspec_grizli', redshift=0)
                spec.unit_conversion('AA', 'FLAM', norm_flux=1e-22)
                valid = True
            except:
                valid = False
                sample_df.loc[idx, 'failed_opening'] = True

            # Valid file
            if valid:



                # Components detection
                spec.infer.components(exclude_continuum=False)
                z_flux_sum, z_pixel_count = spec.fit.redshift(lines_df, band_vsigma=bands_vsigma, z_min=z_min, z_max=z_max, delta_z=z_step, map_min_R=map_min_R,
                                                              plot_results=True)
                redshift_flag = flag_measurement(z_flux_sum, z_pixel_count)
                print()

                spec.plot.spectrum(show_components=True, ax_cfg={'title': f'z = {z_flux_sum}, {z_pixel_count}. Flag = {redshift_flag}'})

#                 # Redshift fitting
#                 z_flux_sum = spec.fit.redshift(lines_df, band_vsigma=bands_vsigma, z_min=z_min, z_max=z_max, delta_z=z_step, mode='key', plot_results=True)
#                 z_pixel_count = spec.fit.redshift(lines_df, band_vsigma=bands_vsigma, z_min=z_min, z_max=z_max, delta_z=z_step, mode='xor')
#                 redshift_flag = flag_measurement(z_flux_sum, z_pixel_count)
#                 print(f' - z = {z_flux_sum}, {z_pixel_count}. Flag = {redshift_flag}')
#
#                 # Save the results
#                 sample_df.loc[idx, 'z_aspect_flux_sum'] = z_flux_sum
#                 sample_df.loc[idx,'z_aspect_pixel_count'] = z_pixel_count
#                 sample_df.loc[idx,'z_aspect_flag'] = redshift_flag
#
#     end_time = np.round((time() - start_time) / 60, 2)
#
#     # Save the redshift measurements
#     sample_fname = f'./aspect_DAWN_prism_control_classifier-v12-RF_min-max-log_12-pixels_sample_v3.csv'
#     lime.save_frame(sample_fname, sample_df)
#     print(f'- Saved measurements file {sample_fname}')
#
#     # Show failing files
#     if np.any(sample_df['failed_opening']):
#         print(f"- Failed opening: {sample_df[sample_df['failed_opening']]}")

# Make the diagnostic plot
# idcs_good = (sample_df.grade == 3).to_numpy()
plot_redshift_diagnostic(sample_df)

