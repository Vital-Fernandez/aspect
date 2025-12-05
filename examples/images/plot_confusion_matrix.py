import numpy as np
import aspect
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt, rc_context
from lime import theme

# Configuration
cfg_file = '../../training/12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])
fig_folder = f'/home/vital/Dropbox/Astrophysics/Tools/aspect'

# Model reference
label = f'aspect_{norm}_{version}'
cfg = sample_cfg[f'randomforest_{version}']
model_address = output_folder/'results'/f'{label}_model.joblib'

# Read the sample files:
y_arr = np.loadtxt(output_folder/f'pred_array_{version}.txt', dtype=str)
data_matrix = np.loadtxt(output_folder/f'data_array_{norm}_{version}.txt', delimiter=',')

# Training the sample
X_train, y_train, X_test, y_test = aspect.trainer.get_training_test_sets(data_matrix, y_arr, 0.1,
                                                                         n_pixel_features=sample_cfg[f'properties_{version}']['box_pixels'], n_scale_features=1)
# Load the model
ml_function = aspect.load_model(model_address)
y_pred = ml_function.predict(X_test)

# Plot confusion matrix
labels = cfg['categories']
y_test = np.vectorize(aspect.cfg['number_shape'].get)(y_test)
y_pred = np.vectorize(aspect.cfg['number_shape'].get)(y_pred)
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig_cfg = {'figure.figsize':(7, 7), 'figure.dpi': 350,
           'axes.labelsize': 12, 'axes.labelpad': 10,
           'xtick.labelsize': 10, 'ytick.labelsize': 10}

with rc_context(theme.fig_defaults(user_fig=fig_cfg)):

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Plot heatmap on the axis
    sns.heatmap(cm_normalized, annot=True, fmt='.0%', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 10})

    # Customize axis labels and title
    ax.set_ylabel('Actual class')
    ax.set_xlabel('Predicted class')
    # ax.set_title('Multiclass Confusion Matrix')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va='center')

    # fig.patch.set_facecolor("#2B2B2B")       # Whole figure
    # ax.set_facecolor("#2B2B2B")
    # ax.tick_params(colors="#CCCCCC")         # Tick labels
    # ax.xaxis.label.set_color("#CCCCCC")      # X-axis label
    # ax.yaxis.label.set_color("#CCCCCC")      # Y-axis label
    # ax.title.set_color("#CCCCCC")            # Title

    # Save and show
    fig.savefig(f'{fig_folder}/aspect_confusion_matrix_{version}_{norm}.png', bbox_inches='tight')
    # plt.show()
