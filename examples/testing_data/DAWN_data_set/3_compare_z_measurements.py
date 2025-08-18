import lime
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, rc_context

# Read sample database
sample_fname = './aspect_DAWN_prism_v3_selection.csv'
sample_df = lime.load_frame(sample_fname)

# Prepare the data
idcs = pd.notnull(sample_df.zfit) & pd.notnull(sample_df.z) & pd.notnull(sample_df.zxor)
idcs = idcs & (np.abs(sample_df['zxor'] - sample_df['zkey']) <= 0.05 * sample_df['zxor'])

z_fit_dawn_arr = sample_df.loc[idcs].z.to_numpy()
z_dawn_arr = sample_df.loc[idcs].zfit.to_numpy()
z_key_arr = sample_df.loc[idcs].zxor.to_numpy()

x_arr = z_fit_dawn_arr
y_arr = z_key_arr

conf_plot = {"figure.dpi": 200,
            "figure.figsize": [5, 5],
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "legend.fontsize": 7,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.size": 5}

with rc_context(lime.theme.fig_defaults(conf_plot)):

    fig, ax = plt.subplots()

    ax.scatter(z_fit_dawn_arr, y_arr, alpha=0.2)

    ax.plot([0, 6.5], [0, 6.5], 'r--', label='x = y')

    # Plot format
    ax.grid(True, which='both', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6.5)

    # Plot working
    ax.set(**{'title': f'Dawn galaxy redshift comparison ({np.sum(idcs)}/{sample_df.index.size})',
              'xlabel': 'Dawn zfit values',
              'ylabel': f'Aspect redshift (key)'})

plt.show()
