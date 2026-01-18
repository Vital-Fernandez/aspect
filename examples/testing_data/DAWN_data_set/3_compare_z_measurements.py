import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context

lime.theme.set_style('dark')

# Read sample database
sample_fname = './aspect_DAWN_prism_v4_measurements.csv'
sample_df = lime.load_frame(sample_fname)

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

with rc_context(lime.theme.fig_defaults(conf_plot)):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)

    for i, z_param in enumerate(z_column):

        # Trendlines
        axes[i].plot([0, z_limit], [0, z_limit], '-', color='yellow', linewidth=0.5)

        # Data
        x, y, w = sample_df.loc[:, z_param].to_numpy(), sample_df.loc[:, 'z'].to_numpy(), sample_df.loc[:, z_other[i]].to_numpy()

        idcs_match = np.isclose(x, y, rtol=0.05)

        idcs_miss = ~idcs_match
        idcs_both_fail = np.isclose(x[idcs_miss], w[idcs_miss], rtol=0.05)
        frac = np.round(idcs_both_fail.sum()/sample_df.index.size * 100).astype(int)
        axes[i].scatter(x[idcs_miss][idcs_both_fail], y[idcs_miss][idcs_both_fail], alpha=0.5, label=f'Both techniques false detection {frac} %', color='tab:red', edgecolors='none')

        frac = np.round(idcs_match.sum()/sample_df.index.size * 100).astype(int)
        axes[i].scatter(x[idcs_match], y[idcs_match], alpha=0.5, label=f'True detection {frac} % ({x_label_list[i]} sum)', color='tab:blue', edgecolors='none')

        frac = np.round((~idcs_both_fail).sum()/sample_df.index.size * 100).astype(int)
        axes[i].scatter(x[idcs_miss][~idcs_both_fail], y[idcs_miss][~idcs_both_fail], alpha=0.2, label=f'False detection {frac} % ({x_label_list[i]} sum)', color='tab:orange', edgecolors='none')

        # Plot format
        axes[i].grid(True, which='both', linewidth=0.5, alpha=0.7)
        axes[i].set_xlim(0, z_limit)
        axes[i].set_ylim(0, z_limit)

        # Plot wording
        axes[i].set_xlabel(r'$z_{Aspect}$' + f' ({x_label_list[i]} sum)')
        if i == 0: axes[i].set_ylabel(r'$z_{true}$ (DAWN archive)')

        axes[i].legend(loc='upper left', framealpha=1)

    fig.suptitle(f"Redshift comparison: {sample_df.index.size} galaxies, 0.37 seconds per object", fontsize=16)

plt.tight_layout()
plt.show()


# 'title': f'Dawn galaxy redshift comparison ({np.sum(idcs)}/{sample_df.index.size})',

# # Trendlines
# axes[0].plot([0, z_limit], [0, z_limit], 'r--', label='x = y')
# axes[1].plot([0, z_limit], [0, z_limit], 'r--', label='x = y')
#
# # Data
# x, y = sample_df.loc[idcs_zor].zkey.to_numpy(), sample_df.loc[idcs_zor].z.to_numpy()
# axes[0].scatter(x, y, alpha=0.2, edgecolors='none')
#
# x, y = sample_df.loc[idcs_zkey].zkey.to_numpy(), sample_df.loc[idcs_zkey].z.to_numpy()
# axes[1].scatter(x, y, alpha=0.2, edgecolors='none')
#
# # Plot format
# axes[0].grid(True, which='both', linewidth=0.5, alpha=0.7)
# axes[1].grid(True, which='both', linewidth=0.5, alpha=0.7)
# axes[0].set_xlim(0, z_limit)
# axes[1].set_xlim(0, z_limit)
# axes[0].set_ylim(0, z_limit)
# axes[1].set_ylim(0, z_limit)
#
# # Plot wording
# axes[0].set(**{'xlabel': f'Aspect redshift (Pixel count)', 'ylabel': f'True redshift (DAWN)'})
# axes[1].set(**{'xlabel': f'Aspect redshift (Flux count)'})


# I have a dataframe where the column 'z' is the true value and columns "zxor" and "zkey" are the predictions from two different techniques. Can you give me the python code to plot the confusion matrix showing the relation between true values and the predicitons