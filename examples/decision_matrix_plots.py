import aspect
import numpy as np
from matplotlib import pyplot as plt, gridspec, colors

from aspect import decision_matrix_plot

# Create a 12x12 array with random 0s and 1s
# np.random.seed(0)  # For reproducibility
# decision_matrix = np.random.randint(2, size=(12, 12))

decision_matrix_plot('choice', "decision_matrix_plot.png", exclude_diagonal=True, show_categories=True)