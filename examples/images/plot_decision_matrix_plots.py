from pathlib import Path
from aspect import decision_matrix_plot

fig_folder = Path(f'/home/vital/Dropbox/Astrophysics/Tools/aspect')
categories = ['white-noise', 'continuum', 'cosmic-ray', 'emission', 'doublet', 'dead-pixel', 'absorption']

conf_plot = {"figure.dpi": 300,
            "figure.figsize": [5, 5],
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "legend.fontsize": 7,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "font.size": 5}

decision_matrix_plot('choice',
                     fig_folder/"decision_matrix_plot.png",
                     categories=categories,
                     exclude_diagonal=True, show_categories=False,
                     cfg_fig=conf_plot)

decision_matrix_plot('time',
                     fig_folder/"time_matrix_plot.png",
                     categories=categories,
                     exclude_diagonal=True, show_categories=False,
                     cfg_fig=conf_plot)