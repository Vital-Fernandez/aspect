import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context

lime.theme.set_style('dark')

# Read sample database
# sample_fname = '/home/vital/PycharmProjects/aspect/examples/testing_data/DAWN_data_set/aspect_DAWN_prism_v3_measurements_randomforest_v1.csv'
# sample_fname = '/home/vital/PycharmProjects/aspect/examples/testing_data/DAWN_data_set/aspect_DAWN_prism_v3_measurements_randomforest_v4.csv'
# sample_fname = '/home/vital/PycharmProjects/aspect/examples/testing_data/DAWN_data_set/aspect_DAWN_prism_v3_measurements_MLP_v2.csv'
# sample_fname = '/home/vital/PycharmProjects/aspect/examples/testing_data/DAWN_data_set/aspect_DAWN_prism_v3_measurements_MLP_v2.csv'
# sample_fname = './aspect_DAWN_prism_classifier_v12_RF_v5_measurements.csv'
# sample_fname = './aspect_DAWN_prism_classifier_v12_RF_v6_measurements.csv'
sample_fname = './aspect_DAWN_prism_classifier_v12_RF_v8_flags1-2-3_selection.csv'

sample_df = lime.load_frame(sample_fname)

name_selection = 'grade 3'
idcs_selection = (sample_df.grade == 3)
sample_df = sample_df.loc[idcs_selection]

if np.any(pd.isnull(sample_df.z)):
    raise KeyError('Null redshift true measurements')

conf_plot = {"figure.dpi": 200,
            "figure.figsize": [10, 5],
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 5}

z_limit = 9
x_label_list = ['Pixel', 'Flux']
z_column = ['zxor', 'zkey']
z_other = ['zkey', 'zxor']

close_mask = np.isclose(sample_df["z"], sample_df["zfit"], rtol=0.05, equal_nan=False)
print(f"Close:     {close_mask.sum()}")
print(f"Not close: {(~close_mask).sum()}")
print(f"Null in A: {sample_df['z'].isnull().sum()}")
print(f"Null in B: {sample_df['zfit'].isnull().sum()}")
'''
Ok I have a dataframe variable called "sample_df", where one column represent the true value and is callled "zfit" and two columsn with the resuls from
two methodologies called "zkey" and "zxor". I want the python code to tell me:
1) number of null entries in each of the true, and methodologies columns and how many are numeric
2) Number of entries both methodologies rows have numerical entries and how many close within 5% proximity and how many are not
3) How many entries each methodology is close within 5% of the true value 
4) How many entries both methodologies are close within 5% of each other but not within 5% of the true value.
5) you give me this values as number count and as a percentage of the total of sample_df

'''

total = len(sample_df.index)

# ── 1) Nulls and numeric counts ───────────────────────────────────────────────
for col in ["zfit", "zkey", "zxor"]:
    n_null    = sample_df[col].isnull().sum()
    n_numeric = total - n_null
    print(f"{col:>6}: {n_null} nulls ({n_null/total:.1%}), {n_numeric} numeric ({n_numeric/total:.1%})")

# ── 2) Both methodologies numeric → close to each other ───────────────────────
both_numeric   = sample_df["zkey"].notna() & sample_df["zxor"].notna()
close_each_other = both_numeric & np.isclose(sample_df["zkey"], sample_df["zxor"], rtol=0.05, equal_nan=False)

n_both_numeric     = both_numeric.sum()
n_close_each_other = close_each_other.sum()
n_not_close        = (both_numeric & ~close_each_other).sum()

print(f"\nBoth numeric:               {n_both_numeric} ({n_both_numeric/total:.1%})")
print(f"  Close to each other:      {n_close_each_other} ({n_close_each_other/total:.1%})")
print(f"  Not close to each other:  {n_not_close} ({n_not_close/total:.1%})")

mask_9 = sample_df["capers_flag"] == 9
sub = sample_df[mask_9]
n9  = len(sub)

# ── 3) Capers flag ───────────────────────────────────
print(f"\n── capers_flag == 9  (n={n9}, {n9/total:.1%} of total) ──")

# Null fractions for zkey, zxor, zfit
for col in ["zkey", "zxor", "zfit"]:
    n_null    = sub[col].isnull().sum()
    n_numeric = n9 - n_null
    print(f"  {col:>6}: {n_null} null ({n_null/n9:.1%}), {n_numeric} numeric ({n_numeric/n9:.1%})")

# Grade breakdown
for grade in [1, 2, 3]:
    n_grade = (sub["grade"] == grade).sum()
    print(f"  grade={grade}: {n_grade} ({n_grade/n9:.1%})")


# ── 3) Each methodology close to true value ───────────────────────────────────
has_true = sample_df["zfit"].notna()

zkey_close_true = has_true & sample_df["zkey"].notna() & np.isclose(sample_df["zkey"], sample_df["zfit"], rtol=0.05, equal_nan=False)
zxor_close_true = has_true & sample_df["zxor"].notna() & np.isclose(sample_df["zxor"], sample_df["zfit"], rtol=0.05, equal_nan=False)

print(f"\nzkey close to zfit: {zkey_close_true.sum()} ({zkey_close_true.sum()/total:.1%})")
print(f"zxor close to zfit: {zxor_close_true.sum()} ({zxor_close_true.sum()/total:.1%})")

# ── 4) Both close to each other but NOT to true value ─────────────────────────
both_close_not_true = close_each_other & ~zkey_close_true & ~zxor_close_true
n_both_close_not_true = both_close_not_true.sum()

print(f"\nBoth close to each other but not to true: {n_both_close_not_true} ({n_both_close_not_true/total:.1%})")

# ── Bias & error magnitude ────────────────────────────────────────────────────
sample_df["err_key"] = sample_df["zkey"] - sample_df["zfit"]
sample_df["err_xor"] = sample_df["zxor"] - sample_df["zfit"]

for col, err in [("zkey", "err_key"), ("zxor", "err_xor")]:
    print(f"{col}  bias={sample_df[err].mean():.4f}  MAE={sample_df[err].abs().mean():.4f}  std={sample_df[err].std():.4f}")

# ── Relative error distribution ───────────────────────────────────────────────
sample_df["rel_key"] = (sample_df["zkey"] - sample_df["zfit"]) / sample_df["zfit"].abs()
sample_df["rel_xor"] = (sample_df["zxor"] - sample_df["zfit"]) / sample_df["zfit"].abs()

print(sample_df[["rel_key", "rel_xor"]].describe(percentiles=[.25, .5, .75, .90, .95]))

# ── 2x2 breakdown: where each method wins ─────────────────────────────────────
key_only  = zkey_close_true & ~zxor_close_true
xor_only  = zxor_close_true & ~zkey_close_true
both_close = zkey_close_true & zxor_close_true
neither    = ~zkey_close_true & ~zxor_close_true

for label, mask in [("key only", key_only), ("xor only", xor_only),
                    ("both",     both_close), ("neither", neither)]:
    print(f"{label:>10}: {mask.sum()} ({mask.sum()/total:.1%})")

# ── Outlier behaviour among failures ─────────────────────────────────────────
for col, rel, close_mask in [("zkey", "rel_key", zkey_close_true),
                              ("zxor", "rel_xor", zxor_close_true)]:
    failures = sample_df.loc[~close_mask, rel]
    print(f"{col} failures — median rel err: {failures.median():.2%}, 95th pct: {failures.abs().quantile(0.95):.2%}")

# ── Bland-Altman plots ────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, method, err in zip(axes, ["zkey", "zxor"], ["err_key", "err_xor"]):
    mean_val = (sample_df[method] + sample_df["zfit"]) / 2
    ax.scatter(mean_val, sample_df[err], alpha=0.3, s=10)
    ax.axhline(sample_df[err].mean(), color="red", label="bias")
    ax.axhline(sample_df[err].mean() + 1.96*sample_df[err].std(), color="orange", linestyle="--", label="±1.96σ")
    ax.axhline(sample_df[err].mean() - 1.96*sample_df[err].std(), color="orange", linestyle="--")
    ax.set_xlabel("Mean of method & truth"); ax.set_ylabel("Difference")
    ax.set_title(f"Bland-Altman: {method}"); ax.legend()
plt.tight_layout()
plt.show()

# with rc_context(lime.theme.fig_defaults(conf_plot)):
#
#     fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
#
#     for i, z_param in enumerate(z_column):
#
#         # Trendlines
#         axes[i].plot([0, z_limit], [0, z_limit], '-', color='yellow', linewidth=0.5)
#
#         # Data
#         x, y = sample_df.loc[:, z_param].to_numpy(), sample_df.loc[:, 'z'].to_numpy()
#         # x, y, w = sample_df.loc[:, z_param].to_numpy(), sample_df.loc[:, 'z'].to_numpy(), sample_df.loc[:, z_other[i]].to_numpy()
#         idcs_match = np.isclose(x, y, rtol=0.05)
#         not_null = np.sum(~pd.isnull(x))
#
#         idcs_miss = ~idcs_match
#         idcs_both_fail = np.isclose(x[idcs_miss], w[idcs_miss], rtol=0.05)
#
#         frac = np.round(idcs_both_fail.sum()/sample_df.index.size * 100).astype(int)
#         x_coord, y_coord = x[idcs_miss][idcs_both_fail], y[idcs_miss][idcs_both_fail]
#         axes[i].scatter(x_coord, y_coord, alpha=0.5, label=f'Both techniques false detection ({frac} %)', color='tab:red', edgecolors='none')
#
#         frac = np.round(idcs_match.sum()/sample_df.index.size * 100).astype(int)
#         x_coord, y_coord = x[idcs_match], y[idcs_match]
#         axes[i].scatter(x_coord, y_coord, alpha=0.5, label=f'True detection {frac} % ({x_label_list[i]} sum)', color='tab:blue', edgecolors='none')
#
#         frac = np.round((~idcs_both_fail).sum()/sample_df.index.size * 100).astype(int)
#         x_coord, y_coord = x[idcs_miss][~idcs_both_fail], y[idcs_miss][~idcs_both_fail]
#         axes[i].scatter(x_coord, y_coord, alpha=0.2, label=f'False detection {frac} % ({x_label_list[i]} sum)', color='tab:orange', edgecolors='none')
#
#         # Plot format
#         axes[i].grid(True, which='both', linewidth=0.5, alpha=0.7)
#         axes[i].set_xlim(0, z_limit)
#         axes[i].set_ylim(0, z_limit)
#
#         # Plot wording
#         axes[i].set_xlabel(r'$z_{Aspect}$' + f' {x_label_list[i]} sum: {not_null} measurements')
#         if i == 0: axes[i].set_ylabel(r'$z_{true}$ (DAWN archive)')
#
#         axes[i].legend(loc='upper left', framealpha=1)
#
#     fig.suptitle(f"DAWNT archive blind redshift comparison: {sample_df.index.size} grade 3 objects", fontsize=16)
#
#     plt.tight_layout()
#     plt.show()
