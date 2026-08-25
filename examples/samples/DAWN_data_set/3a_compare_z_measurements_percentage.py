import lime
import numpy as np
from matplotlib import pyplot as plt, rc_context


def plot_redshift_diagnostic(sample_df, output_address=None, z_true_hdr='z',
                             flag_hdr='z_aspect_flag', idcs_good=None,
                             z_pixel_hdr='z_aspect_pixel_count',
                             z_flux_hdr='z_aspect_flux_sum',
                             z_limit=12, conf_plot=None, rtol_diag=0.05):

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
                     "legend.fontsize": 11, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "font.size": 5}

    x_label_list = ['pixel count', 'flux sum']
    z_column = [z_pixel_hdr, z_flux_hdr]
    z_other = [z_flux_hdr, z_pixel_hdr]

    flag_col = sample_df[flag_hdr]
    flag_3 = (flag_col == 3).to_numpy()
    flag_2 = (flag_col == 2).to_numpy()
    flag_1 = (flag_col == 1).to_numpy()
    flag_0 = (flag_col == 0).to_numpy()

    # Missing-measurement flag per technique: panel 0 (pixel) -> flag 0 ("no lines"),
    # panel 1 (flux) -> flag 1 ("1 line"). Kept as a list so each panel can also plot
    # the *other* technique's false negatives for direct comparison.
    missing_flags = [flag_0, flag_1]
    fn_colors = ['tab:green', 'tab:purple']
    fn_labels = ['No lines detected (flag 0)', '1 line detected (flag 1)']

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
            idcs_TP = np.isclose(x, y, rtol=rtol_diag, equal_nan=False)

            # 1 True positive
            frac = pct(idcs_TP.sum(), n_good)
            ax.scatter(x[idcs_TP], y[idcs_TP], alpha=0.5,
                       label=f'True positive: {frac}% of {n_good}',
                       color='tab:blue', edgecolors='none')

            # 2 False positive (single technique)
            idcs_FP2 = (flag_2) & ~np.isclose(x, y, rtol=rtol_diag, equal_nan=False)
            frac = pct(idcs_FP2.sum(), n_good)
            ax.scatter(x[idcs_FP2], y[idcs_FP2], alpha=0.2,
                       label=f'False positive: {frac}% of {n_good}',
                       color='tab:orange', edgecolors='none')

            # 3 False positive (both techniques)
            idcs_FP3 = flag_3 & ~np.isclose(x, y, rtol=rtol_diag, equal_nan=False)
            frac = pct(idcs_FP3.sum(), n_good)
            ax.scatter(x[idcs_FP3], y[idcs_FP3], alpha=0.2,
                       label=f'False positive both techniques: {frac}% of {n_good}',
                       color='tab:red', edgecolors='none')

            # 4 True negatives
            idcs_TN = idcs_bad & ~(flag_2 | flag_3)
            frac = pct(idcs_TN.sum(), n_bad)
            ax.scatter([], [], alpha=0.8, label=f'True negative: {frac}% of {n_bad}',
                       color='tab:grey', edgecolors='none', marker='s')

            # 5 False negatives -> both distributions now share the same right-hand axis:
            #    - this panel's own technique, in its native color
            #    - the OTHER technique's false negatives, overlaid in orange for comparison
            own_idx, other_idx = i, 1 - i

            idcs_FN_own = idcs_good & missing_flags[own_idx]
            z_arr_own = sample_df.loc[idcs_FN_own, z_true_hdr].to_numpy()
            frac_own = pct(idcs_FN_own.sum(), n_good)
            own_color = fn_colors[own_idx]

            idcs_FN_other = idcs_good & missing_flags[other_idx]
            z_arr_other = sample_df.loc[idcs_FN_other, z_true_hdr].to_numpy()
            frac_other = pct(idcs_FN_other.sum(), n_good)

            # Own technique's false negatives (solid fill, native color)
            ax_histy.hist(z_arr_own, bins=z_bins, orientation='horizontal',
                          color=own_color, alpha=0.7)
            #
            # # Other technique's false negatives (orange, hatched outline so it
            # # reads as a separate overlaid series rather than a stack)
            # ax_histy.hist(z_arr_other, bins=z_bins, orientation='horizontal',
            #               color='tab:orange', alpha=0.2, histtype='stepfilled',
            #               edgecolor='tab:orange', linewidth=1.0)

            # Legend proxy for the own-technique false negatives only; the other
            # technique's false negatives are shown in the histy overlay but reuse
            # the same orange already labeled by the "False positive" entry above,
            # so no separate legend entry is added for it.
            ax.scatter([], [], alpha=0.7, marker='s', edgecolors='none', color=own_color,
                       label=fn_labels[own_idx])#f'No lines detected (flag 0{fn_labels[own_idx]}): '    f'{frac_own}% of {n_good}')

            # Scatter plot formatting
            ax.grid(True, which='both', linewidth=0.5, alpha=0.7)
            ax.set_xlim(0, z_limit)
            ax.set_ylim(0, z_limit)
            ax.set_xlabel(r'$z_{Aspect}$' + f' {x_label_list[i]}')
            if i == 0:
                ax.set_ylabel(r'$z_{true}$')
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.legend(loc='upper left', framealpha=0.8)

            # Histogram formatting
            ax_histy.tick_params(axis='y', labelleft=False)

        fig.suptitle(f"DAWN archive blind redshift comparison: {n_rows} spectra", fontsize=16)
        plt.tight_layout()

        if output_address is None:
            plt.show()
        else:
            plt.savefig(output_address)

    return fig

sample_fname = f'./aspect_DAWN_prism_control_classifier-v12-RF_min-max-log_12-pixels_sample_v34.csv'
sample_df = lime.load_frame(sample_fname)
sample_df = sample_df.loc[sample_df.z > 0.1]

idcs_good = (sample_df.grade == 3).to_numpy()
plot_redshift_diagnostic(sample_df, output_address='DAWN_diagnostic_v36.png', idcs_good=idcs_good)