import lime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, rc_context

C_KMS = 299792.458

# Default success criteria used in the comparison. rtol_aspect is the one you are
# most likely to want to change (e.g. try 0.03 or 0.10 instead of the paper's 5%);
# it is exposed as an argument on every function below rather than hardcoded here.
RTOL_ASPECT_DEFAULT = 0.05  # This paper: |z_aspect - z_true| <= rtol_aspect * z_true
DELTA_FP19_DEFAULT = 0.003  # Frontera-Pons et al. 2019: failure if |z_est - z_true| > delta_fp19 * (1 + z_true)
AGREE_FP19_DEFAULT = 0.003  # Frontera-Pons et al. 2019, Algorithm 3: the two estimators agree if |z_1 - z_2| <= agree_fp19
BIN_S20_DEFAULT = 0.001     # Stivaktakis et al. 2020: correct if same bin_s20-wide class


def _nmad(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else 1.4826 * np.median(np.abs(x - np.median(x)))


def _rate(mask, n):
    return np.nan if n == 0 else mask.sum() / n


def compute_redshift_precision(sample_df, idcs_good=None, z_true_hdr='z', flag_hdr='z_aspect_flag',
                               z_pixel_hdr='z_aspect_pixel_count', z_flux_hdr='z_aspect_flux_sum',
                               z_limit=12, z_bin_width=1.0,
                               rtol_aspect=RTOL_ASPECT_DEFAULT, delta_fp19=DELTA_FP19_DEFAULT,
                               agree_fp19=AGREE_FP19_DEFAULT, bin_s20=BIN_S20_DEFAULT):

    """Precision diagnostics of the blind redshift measurements in the format of
    Frontera-Pons et al. (2019) and Stivaktakis et al. (2020).

    All the statistics are restricted to the good objects (``idcs_good``), since the
    dispersion against a tentative reference redshift is not meaningful. Within them,
    the "measured" objects are those with a finite prediction for the technique.

    Parameters
    ----------
    sample_df : pandas.DataFrame
        Table with the true redshift, the two aspect predictions and the flag.
    idcs_good : array-like of bool, optional
        Boolean mask selecting the objects with a reliable reference redshift.
        If not given, every object is treated as good.
    z_true_hdr, flag_hdr, z_pixel_hdr, z_flux_hdr : str, optional
        Column names, same defaults as in plot_redshift_diagnostic.
    z_limit : float, optional
        Upper limit of the redshift bins (default 12).
    z_bin_width : float, optional
        Width of the redshift bins for the outlier rates (default 1.0).
    rtol_aspect : float, optional
        Relative tolerance of this paper's success criterion,
        |z_aspect - z_true| <= rtol_aspect * z_true (default 0.05, i.e. 5%).
        This is the number to change to test a different precision threshold;
        it matches the ``rtol`` used in ``np.isclose`` in the plotting script.
    delta_fp19 : float, optional
        Frontera-Pons et al. (2019) catastrophic-outlier threshold on
        |z_est - z_true| / (1 + z_true) (default 0.003).
    agree_fp19 : float, optional
        Frontera-Pons et al. (2019) agreement threshold between the two
        estimators, |z_1 - z_2| (default 0.003).
    bin_s20 : float, optional
        Stivaktakis et al. (2020) classification bin width in z (default 0.001);
        a measurement is correct if |z_pred - z_true| <= bin_s20 / 2.

    Returns
    -------
    tech_df : pandas.DataFrame
        One row per technique: success rates at the three criteria, bias and scatter.
    combo_df : pandas.DataFrame
        Combination of the two techniques (Frontera-Pons et al. 2019, Table 1 and Sect. 6.5).
    bin_df : pandas.DataFrame
        Outlier rate per technique and redshift bin (Frontera-Pons et al. 2019, Fig. 8).
    residuals : dict
        Per technique arrays (z_true, dz, delta, ok_aspect, ok_fp19) of the measured
        good objects, for the plots.
    """

    n_rows = sample_df.index.size
    idcs_good = np.ones(n_rows, dtype=bool) if idcs_good is None else np.asarray(idcs_good, dtype=bool)
    n_good = int(idcs_good.sum())

    # Force numeric arrays (None / 'nan' strings become NaN)
    z_true = pd.to_numeric(sample_df[z_true_hdr], errors='coerce').to_numpy(dtype=float)
    flag = pd.to_numeric(sample_df[flag_hdr], errors='coerce').to_numpy(dtype=float)
    z_tech = {'pixel count': pd.to_numeric(sample_df[z_pixel_hdr], errors='coerce').to_numpy(dtype=float),
              'flux sum': pd.to_numeric(sample_df[z_flux_hdr], errors='coerce').to_numpy(dtype=float)}

    z_bins = np.arange(0, z_limit + z_bin_width, z_bin_width)

    # ---------------------------------------------------------------- per technique
    tech_rows, bin_rows, residuals = [], [], {}
    has, ok5_all, ok3_all = {}, {}, {}

    for label, z_fit in z_tech.items():

        idcs_z = idcs_good & np.isfinite(z_fit)
        n_z = int(idcs_z.sum())
        zt, zf = z_true[idcs_z], z_fit[idcs_z]

        dz = zf - zt
        delta = dz / (1 + zt)

        ok_aspect = np.isclose(zf, zt, rtol=rtol_aspect)      # identical logic to the plotting script
        ok_fp19 = np.abs(delta) <= delta_fp19
        ok_s20 = np.abs(dz) <= bin_s20 / 2

        bias = np.median(delta) if n_z else np.nan
        nmad = _nmad(delta)

        tech_rows.append({'technique': label,
                          'N_good': n_good,
                          'N_measured': n_z,
                          'null_return_rate': 1 - n_z / n_good if n_good else np.nan,
                          # This paper (|dz| <= rtol_aspect * z_true)
                          f'TP_rate_{rtol_aspect:g} (over good)': _rate(ok_aspect, n_good),
                          f'FP_rate_{rtol_aspect:g} (over good)': _rate(~ok_aspect, n_good),
                          f'success_{rtol_aspect:g} (over measured)': _rate(ok_aspect, n_z),
                          # Frontera-Pons et al. 2019 (|dz| <= delta_fp19 (1 + z_true))
                          'success_FP19 (over good)': _rate(ok_fp19, n_good),
                          'success_FP19 (over measured)': _rate(ok_fp19, n_z),
                          # Stivaktakis et al. 2020 (same bin_s20 class)
                          'success_S20 (over measured)': _rate(ok_s20, n_z),
                          # Precision of the normalized residual delta = dz / (1 + z_true)
                          'bias_delta (median)': bias,
                          'sigma_NMAD_delta': nmad,
                          'sigma_NMAD_kms': nmad * C_KMS,
                          f'sigma_delta_TP (std, {rtol_aspect:g})': np.std(delta[ok_aspect]) if ok_aspect.sum() > 1 else np.nan,
                          f'sigma_dz_TP (std, {rtol_aspect:g})': np.std(dz[ok_aspect]) if ok_aspect.sum() > 1 else np.nan,
                          'sigma_dz_FP19 (std, non-outliers)': np.std(dz[ok_fp19]) if ok_fp19.sum() > 1 else np.nan})

        # Outlier rate per redshift bin (Frontera-Pons et al. 2019, Fig. 8)
        for z_lo, z_hi in zip(z_bins[:-1], z_bins[1:]):
            in_bin = (zt >= z_lo) & (zt < z_hi)
            n_bin = int(in_bin.sum())
            bin_rows.append({'technique': label, 'z_low': z_lo, 'z_high': z_hi, 'N': n_bin,
                             f'outlier_rate_{rtol_aspect:g}': np.nan if n_bin == 0 else 1 - ok_aspect[in_bin].mean(),
                             'outlier_rate_FP19': np.nan if n_bin == 0 else 1 - ok_fp19[in_bin].mean()})

        residuals[label] = dict(z_true=zt, dz=dz, delta=delta, ok_aspect=ok_aspect, ok_fp19=ok_fp19)

        # Full-length masks for the combination
        has[label] = idcs_z
        ok5_all[label] = np.zeros(n_rows, dtype=bool)
        ok5_all[label][idcs_z] = ok_aspect
        ok3_all[label] = np.zeros(n_rows, dtype=bool)
        ok3_all[label][idcs_z] = ok_fp19

    tech_df = pd.DataFrame(tech_rows).set_index('technique')
    bin_df = pd.DataFrame(bin_rows)

    # ------------------------------------------------- combination of the two techniques
    z_pix, z_flx = z_tech['pixel count'], z_tech['flux sum']
    both = has['pixel count'] & has['flux sum']
    n_both = int(both.sum())

    agreement = {f'agree_FP19 (|z_pix - z_flux| <= {agree_fp19:g})': both & (np.abs(z_pix - z_flx) <= agree_fp19),
                 'agree_flag3 (z_aspect_flag == 3)': both & (flag == 3)}

    combo_rows = []
    for crit_label, ok in ((f'{rtol_aspect:g}', ok5_all), ('FP19', ok3_all)):

        ok_p, ok_f = ok['pixel count'] & both, ok['flux sum'] & both
        row = {'criterion': crit_label,
               'N_good': n_good, 'N_both_measured': n_both,
               'pixel yes / flux yes': int((ok_p & ok_f).sum()),
               'pixel yes / flux no': int((ok_p & ~ok_f).sum()),
               'pixel no / flux yes': int((~ok_p & ok_f).sum()),
               'pixel no / flux no': int((both & ~ok_p & ~ok_f).sum()),
               'success_any (over good)': _rate(ok_p | ok_f, n_good),
               'success_any (over both)': _rate(ok_p | ok_f, n_both)}

        # Robust sub-sample from the agreement of the two estimators (their Sect. 6.5);
        # the flux-sum value is evaluated inside the agreeing sub-sample.
        for agree_label, agree in agreement.items():
            n_agree = int(agree.sum())
            row[f'{agree_label}: fraction of good'] = _rate(agree, n_good)
            row[f'{agree_label}: fraction of both'] = _rate(agree, n_both)
            row[f'{agree_label}: outlier rate'] = np.nan if n_agree == 0 else 1 - _rate(ok_f & agree, n_agree)

        combo_rows.append(row)

    combo_df = pd.DataFrame(combo_rows).set_index('criterion')

    return tech_df, combo_df, bin_df, residuals


def print_summary(tech_df, combo_df, rtol_aspect=RTOL_ASPECT_DEFAULT, delta_fp19=DELTA_FP19_DEFAULT):

    """Console summary in the layout of Frontera-Pons et al. (2019), Table 1."""

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 30)

    print('\nPer technique statistics (good objects):')
    print(tech_df.T.to_string(float_format=lambda v: f'{v:.4f}'))

    n_good = int(tech_df['N_good'].iloc[0])
    crit_rtol = f'{rtol_aspect:g}'
    for crit, desc in ((crit_rtol, f'|dz| <= {rtol_aspect:g} z_true (this paper)'),
                       ('FP19', f'|dz| <= {delta_fp19:g} (1 + z_true) (Frontera-Pons et al. 2019)')):
        row = combo_df.loc[crit]
        print(f'\nSuccess criterion: {desc}; N_good = {n_good}, with both measurements = {int(row["N_both_measured"])}')
        for tech in tech_df.index:
            key = f'TP_rate_{rtol_aspect:g} (over good)' if crit == crit_rtol else 'success_FP19 (over good)'
            n_ok = int(round(tech_df.loc[tech, key] * n_good))
            print(f'  Total {tech} success: {n_ok} ({100 * tech_df.loc[tech, key]:.1f}%)')
        n_any = int(round(row['success_any (over good)'] * n_good))
        print(f'  Total either technique success: {n_any} ({100 * row["success_any (over good)"]:.1f}%)')
        print('  pixel   flux      N')
        for key in ('pixel yes / flux yes', 'pixel yes / flux no', 'pixel no / flux yes', 'pixel no / flux no'):
            p, f = key.split(' / ')
            print(f'  {p.split()[1]:<7} {f.split()[1]:<7} {int(row[key]):>6}')
        for agree in row.index:
            if agree.endswith(': fraction of good'):
                base = agree[:-len(': fraction of good')]
                print(f'  {base}: {100 * row[agree]:.1f}% of good, '
                     f'outlier rate {100 * row[f"{base}: outlier rate"]:.2f}%')


def plot_redshift_precision(residuals, bin_df, tech_df, output_address=None, z_limit=12, conf_plot=None,
                            rtol_aspect=RTOL_ASPECT_DEFAULT, delta_fp19=DELTA_FP19_DEFAULT):

    """Precision figure: residual histograms (Frontera-Pons et al. 2019, Fig. 9; Stivaktakis et al.
    2020, Fig. 11) and outlier rate per redshift bin (Frontera-Pons et al. 2019, Fig. 8)."""

    if conf_plot is None:
        conf_plot = {"figure.dpi": 300, "figure.figsize": [14, 9],
                     "axes.titlesize": 13, "axes.labelsize": 13,
                     "legend.fontsize": 10, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "font.size": 5}

    colors = {'pixel count': 'tab:blue', 'flux sum': 'tab:orange'}
    delta_bins = np.linspace(-rtol_aspect, rtol_aspect, 51)
    dz_bins = np.arange(-z_limit, z_limit + 0.1, 0.1)
    col_rtol = f'outlier_rate_{rtol_aspect:g}'

    with rc_context(lime.theme.fig_defaults(conf_plot)):

        mosaic = [['hist_delta', 'hist_dz'], ['out_rtol', 'out_fp19']]
        fig, axs = plt.subplot_mosaic(mosaic)

        # Normalized residual of the true positives (Frontera-Pons et al. 2019, Fig. 9 style)
        ax = axs['hist_delta']
        for label, res in residuals.items():
            delta_tp = res['delta'][res['ok_aspect']]
            nmad, bias = tech_df.loc[label, 'sigma_NMAD_delta'], tech_df.loc[label, 'bias_delta (median)']
            ax.hist(delta_tp, bins=delta_bins, color=colors[label], alpha=0.5,
                    label=fr'{label}: $\sigma_{{NMAD}}$ = {nmad:.4f}, bias = {bias:+.4f}')
        for x in (-delta_fp19, delta_fp19):
            ax.axvline(x, color='black', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel(r'$(z_{Aspect} - z_{true}) / (1 + z_{true})$')
        ax.set_ylabel('N')
        ax.set_title(f'True positives (|$\\Delta z$| $\\leq$ {rtol_aspect:g} $z_{{true}}$); '
                     f'dashed: $\\pm${delta_fp19:g} threshold')
        ax.legend(loc='upper left', framealpha=0.8)

        # Full residual distribution in log scale (Stivaktakis et al. 2020, Fig. 11 style)
        ax = axs['hist_dz']
        for label, res in residuals.items():
            ax.hist(res['dz'], bins=dz_bins, color=colors[label], alpha=0.5, label=label)
        ax.set_yscale('log')
        ax.set_xlabel(r'$z_{Aspect} - z_{true}$')
        ax.set_ylabel('N (log scale)')
        ax.set_title('All measured objects')
        ax.legend(loc='upper left', framealpha=0.8)

        # Outlier rate per redshift bin (Frontera-Pons et al. 2019, Fig. 8 style)
        for ax_key, col, title in (('out_rtol', col_rtol,
                                    f'Outlier rate, |$\\Delta z$| > {rtol_aspect:g} $z_{{true}}$'),
                                   ('out_fp19', 'outlier_rate_FP19',
                                    f'Outlier rate, |$\\Delta z$| > {delta_fp19:g} (1 + $z_{{true}}$)')):
            ax = axs[ax_key]
            for j, label in enumerate(residuals):
                rows = bin_df.loc[bin_df.technique == label]
                width = (rows.z_high - rows.z_low).to_numpy()
                centers = rows.z_low.to_numpy() + width * (0.3 + 0.4 * j)
                ax.bar(centers, rows[col].to_numpy(), width=0.4 * width, color=colors[label],
                       alpha=0.7, label=label)
                for xc, n_obj, yv in zip(centers, rows.N.to_numpy(), rows[col].to_numpy()):
                    if n_obj > 0:
                        ax.text(xc, min(yv, 0.95) + 0.02, f'{n_obj}', ha='center', va='bottom', fontsize=6)
            ax.set_xlim(0, z_limit)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel(r'$z_{true}$')
            ax.set_ylabel('Outlier rate')
            ax.set_title(title)
            ax.grid(True, which='both', linewidth=0.5, alpha=0.7)
            ax.legend(loc='upper left', framealpha=0.8)

        fig.suptitle('DAWN archive blind redshift precision', fontsize=15)
        plt.tight_layout()

        if output_address is None:
            plt.show()
        else:
            plt.savefig(output_address)

    return fig


# --------------------------------------------------------------------------- run
# Change this single value to test a different ASPECT success tolerance
# (it was 0.05 = 5% throughout the discussion section):
RTOL_ASPECT = 0.05

sample_fname = f'./aspect_DAWN_prism_control_classifier-v12-RF_min-max-log_12-pixels_sample_v34.csv'
sample_df = lime.load_frame(sample_fname)
sample_df = sample_df.loc[sample_df.z > 0.1]

idcs_good = (sample_df.grade == 3).to_numpy()

tech_df, combo_df, bin_df, residuals = compute_redshift_precision(sample_df, idcs_good=idcs_good,
                                                                   rtol_aspect=RTOL_ASPECT)
print_summary(tech_df, combo_df, rtol_aspect=RTOL_ASPECT)

tech_df.to_csv('DAWN_precision_v35_techniques.csv')
combo_df.to_csv('DAWN_precision_v35_combination.csv')
bin_df.to_csv('DAWN_precision_v35_zbins.csv', index=False)
plot_redshift_precision(residuals, bin_df, tech_df, output_address='DAWN_precision_v35.png',
                        rtol_aspect=RTOL_ASPECT)