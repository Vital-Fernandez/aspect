import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context

lime.theme.set_style('dark')

# Read sample database
sample_fname = './aspect_DAWN_prism_classifier_v12_RF_v8_flags1-2-3_selection.csv'
sample_df = lime.load_frame(sample_fname)
name_selection = 'all grade'

# Set the grade 1 to nan
idcs_selection = (sample_df.grade == 1)
sample_df.loc[idcs_selection, 'z'] = np.nan

# name_selection = 'grade 1'
# idcs_selection = (sample_df.grade == 1)
# sample_df = sample_df.loc[idcs_selection]

# if np.any(pd.isnull(sample_df.z)):
#     raise KeyError('Null redshift true measurements')

conf_plot = {"figure.dpi": 200,
            "figure.figsize": [12, 6],
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "legend.fontsize": 10,
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

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

for i, z_param in enumerate(z_column):
    x = sample_df[z_param].to_numpy()
    y = sample_df['z'].to_numpy()
    w = sample_df[z_other[i]].to_numpy()

    not_null   = ~pd.isnull(x)
    not_null_w = ~pd.isnull(w)

    idcs_match     = not_null & np.isclose(x, y, rtol=0.05, equal_nan=False)
    idcs_miss      = not_null & ~idcs_match
    idcs_both_fail = not_null_w[idcs_miss] & np.isclose(x[idcs_miss], w[idcs_miss], rtol=0.05, equal_nan=False)
    meth_null      = pd.isnull(x)

    n_total = sample_df.index.size
    counts = {
        "True detection":              idcs_match.sum(),
        "False detection":             (idcs_miss & ~idcs_both_fail).sum(),  # adjust for indexing
        "Both techniques\nfalse":      idcs_both_fail.sum(),
        "No measurement":              meth_null.sum(),
    }

    # recompute false detection correctly
    false_only = idcs_miss.sum() - idcs_both_fail.sum()
    counts["False detection"] = false_only

    labels = list(counts.keys())
    values = np.array(list(counts.values()))
    fracs  = values / n_total * 100

    matrix = values.reshape(2, 2)
    frac_matrix = fracs.reshape(2, 2)

    row_labels = ["Numeric", "Null"]
    col_labels = ["Correct", "Incorrect"]

    im = axes[i].imshow(matrix, cmap="Blues")
    axes[i].set_xticks([0, 1]); axes[i].set_xticklabels(col_labels)
    axes[i].set_yticks([0, 1]); axes[i].set_yticklabels(row_labels)
    axes[i].set_title(z_param)

    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, f"{matrix[r, c]}\n({frac_matrix[r, c]:.1f}%)",
                         ha='center', va='center', fontsize=11,
                         color='white' if matrix[r, c] > matrix.max() * 0.6 else 'black')

fig.suptitle("Redshift recovery confusion matrix", fontsize=14)
plt.tight_layout()
plt.show()


