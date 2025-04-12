import numpy as np
import aspect
from pathlib import Path
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

# Configuration
cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)
version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

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



print(cm)
print(cm.shape)

# plt.figure(figsize=(7,7))
# # sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
# sns.heatmap(cm_normalized, annot=True,  fmt='.0%', cbar=False, xticklabels=labels, yticklabels=labels)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Multiclass Confusion Matrix')
# plt.savefig(f'{output_folder}/results/{label}_confusion_matrix.png')
# plt.show()

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

# Plot heatmap on the axis
sns.heatmap(cm_normalized, annot=True, fmt='.0%', cbar=False,
            xticklabels=labels, yticklabels=labels, ax=ax)

# Customize axis labels and title
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
ax.set_title('Multiclass Confusion Matrix')

fig.patch.set_facecolor("#2B2B2B")       # Whole figure
ax.set_facecolor("#2B2B2B")
ax.tick_params(colors="#CCCCCC")         # Tick labels
ax.xaxis.label.set_color("#CCCCCC")      # X-axis label
ax.yaxis.label.set_color("#CCCCCC")      # Y-axis label
ax.title.set_color("#CCCCCC")            # Title


# Save and show
fig.savefig(f'{output_folder}/results/{label}_confusion_matrix.png', bbox_inches='tight')
plt.show()


# Change background color
          # Plot area background

# # Change axes colors (ticks, spines, labels)
# ax.tick_params(colors='white')         # Tick labels
# ax.xaxis.label.set_color('white')      # X-axis label
# ax.yaxis.label.set_color('white')      # Y-axis label
# ax.title.set_color('white')            # Title
#
# # Change axis spine colors (borders)
# for spine in ax.spines.values():
#     spine.set_color('white')

# # Read the sample files:
# y_arr = np.loadtxt(output_folder/f'pred_array_doublet.txt', dtype=str)
# data_matrix = np.loadtxt(output_folder/f'data_array_doublet.txt', delimiter=',')
#
# # Plot sample
# n_points = 2500
# shape_list = ['doublet']
# sample_plotter = aspect.plots.CheckSample(data_matrix, y_arr, sample_size=n_points, categories=shape_list, dtype='doublet')
# sample_plotter.show()
