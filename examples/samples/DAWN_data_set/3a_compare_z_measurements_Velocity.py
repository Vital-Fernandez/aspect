import lime
import numpy as np
from matplotlib import pyplot as plt, rc_context

C_KMS = 299792.458


def z_agreement(z_fit, z_true, v_tol=None, n_sigma=3, sigma_kms=140, R=None, dz_step=None):

    """Redshift agreement criterion based on the line bands model (Sect. 3).

    Two redshifts agree when the line band computed at ``z_fit`` contains the transition
    at ``z_true``, so that a subsequent profile fit converges on the correct feature. The
    comparison is done in velocity, |c (z_fit - z_true) / (1 + z_true)| <= v_tol, which is
    the natural unit of a search whose precision is set by the instrument resolving power.

    Parameters
    ----------
    z_fit, z_true : array-like
        Measured and reference redshifts (NaN entries never agree).
    v_tol : float or array-like, optional
        Velocity tolerance in km/s. If ``None`` (default) it is derived from the band model:
        ``n_sigma * sqrt(sigma_kms**2 + sigma_instr**2)``, with
        ``sigma_instr = c / (2 sqrt(2 ln 2) R)`` (Eqs. 5 and 6).
    n_sigma : float, optional
        Number of sigmas of the band half-width (default 3, as in Eq. 8).
    sigma_kms : float, optional
        Characteristic velocity dispersion of the source (default 140 km/s, as in the
        redshift search).
    R : float or array-like, optional
        Resolving power at the lines. If ``None`` it is estimated from the redshift step of
        the search, ``R = 1 / (2 dz_step)`` (Eq. 7).
    dz_step : array-like, optional
        Redshift step of the search per spectrum, median(dlambda) / median(lambda).

    Returns
    -------
    agree : numpy.ndarray of bool
    v_tol : float or numpy.ndarray
        Tolerance actually used (km/s).
    """

    z_fit = np.asarray(z_fit, dtype=float)
    z_true = np.asarray(z_true, dtype=float)
    dv = C_KMS * (z_fit - z_true) / (1 + z_true)

    if v_tol is None:
        if R is None:
            if dz_step is None:
                raise ValueError('Provide a velocity tolerance (v_tol), a resolving power (R) '
                                 'or the redshift step of the search (dz_step).')
            R = 1 / (2 * np.asarray(dz_step, dtype=float))
        sigma_instr = C_KMS / (np.asarray(R, dtype=float) * 2 * np.sqrt(2 * np.log(2)))
        v_tol = n_sigma * np.sqrt(sigma_kms ** 2 + sigma_instr ** 2)

    with np.errstate(invalid='ignore'):
        agree = np.abs(dv) <= v_tol

    return agree, v_tol


def plot_redshift_diagnostic(sample_df, output_address=None, z_true_hdr='z',
                             flag_hdr='z_aspect_flag', idcs_good=None,
                             z_pixel_hdr='z_aspect_pixel_count',
                             z_flux_hdr='z_aspect_flux_sum',
                             z_limit=12, conf_plot=None,
                             v_tol=None, n_sigma=3, sigma_kms=140, R=None, dz_step_hdr=None):

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
    v_tol, n_sigma, sigma_kms, R : optional
        Agreement criterion, see ``z_agreement``. Give either a fixed velocity
        tolerance (``v_tol``, km/s), a resolving power for the band model (``R``),
        or the column with the redshift step of each search (``dz_step_hdr``).
    dz_step_hdr : str, optional
        Column name with the redshift step of the search, median(dlambda)/median(lambda),
        used to estimate R per spectrum when ``R`` and ``v_tol`` are not given.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """

    n_rows = sample_df.index.size

    # Optional indexes to establish the high | low quality redshift measurements in the true sample
    idcs_good = np.ones(n_rows, dtype=bool) if idcs_good is None else np.asarray(idcs_good, dtype=bool)
    idcs_bad = ~idcs_good

    # Redshift step of the search (per spectrum), if available, for the band-model tolerance
    dz_step = sample_df[dz_step_hdr].to_numpy(dtype=float) if dz_step_hdr is not None else None

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

        v_label = None
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

            # Agreement with the reference redshift (velocity criterion from the bands model)
            agree, v_used = z_agreement(x, y, v_tol=v_tol, n_sigma=n_sigma, sigma_kms=sigma_kms,
                                        R=R, dz_step=dz_step)
            v_label = np.nanmedian(v_used)

            # 1 True positive (good objects only, so the percentages close with the FP and FN ones)
            idcs_TP = idcs_good & agree
            frac = pct(idcs_TP.sum(), n_good)
            ax.scatter(x[idcs_TP], y[idcs_TP], alpha=0.5,
                       label=f'True positive (|$\\Delta v$| $\\leq$ {v_label:.0f} km/s): {frac}% of {n_good}',
                       color='tab:blue', edgecolors='none')

            # 2 False positive (single technique)
            idcs_FP2 = idcs_good & flag_2 & ~agree
            frac = pct(idcs_FP2.sum(), n_good)
            ax.scatter(x[idcs_FP2], y[idcs_FP2], alpha=0.2,
                       label=f'False positive: {frac}% of {n_good}',
                       color='tab:orange', edgecolors='none')

            # 3 False positive (both techniques)
            idcs_FP3 = idcs_good & flag_3 & ~agree
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
                       label=f'{fn_labels[own_idx]}: {frac_own}% of {n_good}')

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

        fig.suptitle(f'DAWN archive blind redshift comparison: {n_rows} spectra '
                     f'(agreement |$\\Delta v$| $\\leq$ {v_label:.0f} km/s)', fontsize=16)
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

# Agreement criterion (choose one):
#   a) fixed velocity tolerance:                     v_tol=2000
#   b) bands model with a representative PRISM R:    R=150, n_sigma=3   (~2600 km/s, below the [O III] doublet separation)
#   c) bands model with the step of each search:     dz_step_hdr='z_aspect_step'  (column to be added by the redshift function)
plot_redshift_diagnostic(sample_df, output_address='DAWN_diagnostic_v36_DeltaV.png', idcs_good=idcs_good,
                         R=50, n_sigma=3)