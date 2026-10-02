"""
Random forest hyper-parameter grid search for the ASPECT classification sample.

Loops over the grid shown in the F1 score figure:

    n_estimators      : 20, 40, 60                (figure rows)
    max_depth         : 4, 6, 8                   (figure columns)
    min_samples_leaf  : 1% ... 15% of the category sample size   (panel x axis)
    min_samples_split : 1% ... 15% of the category sample size   (panel y axis)

Results are appended to a csv file one row at a time, so the run can be
interrupted and resumed (already computed combinations are skipped) and the
figure can be produced from a partial grid.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import aspect
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- Configuration

cfg_file = '12_pixels.toml'
sample_cfg = aspect.load_cfg(cfg_file)

version = sample_cfg['meta']['version']
norm = sample_cfg['meta']['scale']
output_folder = Path(sample_cfg['meta']['results_folder'])

rf_cfg = sample_cfg[f'randomforest_{version}']

# Grid (same values as the figure)
n_estimators_arr = np.array([20, 40, 60])
max_depth_arr = np.array([4, 6, 8])
percent_arr = np.array([1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])  # leaf and split axes

# Sample size used for the scan (None -> use the value on the configuration file)
sample_size = 200000
test_fraction = rf_cfg['test_sample_size_fraction']
random_state = rf_cfg['estimator_params'].get('random_state', 42)
max_features = rf_cfg['estimator_params'].get('max_features', 'sqrt')
n_jobs = rf_cfg['estimator_params'].get('n_jobs', -1)

grid_file = output_folder/f'rf_f1_grid_{norm}_{version}.csv'

# ---------------------------------------------------------------- Input sample

category_path = output_folder/f'pred_array_{norm}_{version}.txt'
data_matrix_path = output_folder/f'data_array_{norm}_{version}.txt'

category_arr = np.loadtxt(category_path).astype(int)
data_matrix = np.loadtxt(data_matrix_path, delimiter=',')

# Crop the sample (stratified) if requested
if (sample_size is not None) and (sample_size < category_arr.size):
    data_matrix, _, category_arr, _ = train_test_split(data_matrix, category_arr,
                                                       train_size=sample_size,
                                                       stratify=category_arr,
                                                       random_state=random_state)

X_train, X_test, y_train, y_test = train_test_split(data_matrix, category_arr,
                                                    test_size=test_fraction,
                                                    stratify=category_arr,
                                                    random_state=random_state)

# Category sample size: the reference for the leaf/split percentages
cat_labels, cat_counts = np.unique(y_train, return_counts=True)
category_size = int(np.median(cat_counts))

print(f'Training sample: {y_train.size} entries, {cat_labels.size} categories')
print(f'Category sample size (median): {category_size}')
print(f'Test sample: {y_test.size} entries')

# ---------------------------------------------------------------- Previous runs

columns = ['n_estimators', 'max_depth', 'leaf_percent', 'split_percent',
           'min_samples_leaf', 'min_samples_split', 'category_size',
           'f1_macro', 'f1_weighted', 'accuracy', 'fit_time']

if grid_file.is_file():
    grid_df = pd.read_csv(grid_file)
    done = set(map(tuple, grid_df[['n_estimators', 'max_depth',
                                   'leaf_percent', 'split_percent']].to_numpy()))
    print(f'Resuming: {len(done)} combinations already on {grid_file.name}')
else:
    pd.DataFrame(columns=columns).to_csv(grid_file, index=False)
    done = set()

# ---------------------------------------------------------------- Grid loop

n_total = n_estimators_arr.size * max_depth_arr.size * percent_arr.size ** 2
i_fit = 0
start_scan = time.time()

for n_estimators in n_estimators_arr:
    for max_depth in max_depth_arr:
        for split_percent in percent_arr:
            for leaf_percent in percent_arr:

                i_fit += 1
                key = (int(n_estimators), int(max_depth),
                       float(leaf_percent), float(split_percent))
                if key in done:
                    continue

                # Percentages of the category sample size -> absolute numbers
                min_samples_leaf = max(1, int(round(leaf_percent / 100 * category_size)))
                min_samples_split = max(2, int(round(split_percent / 100 * category_size)))

                estimator = RandomForestClassifier(n_estimators=int(n_estimators),
                                                   max_depth=int(max_depth),
                                                   min_samples_leaf=min_samples_leaf,
                                                   min_samples_split=min_samples_split,
                                                   max_features=max_features,
                                                   random_state=random_state,
                                                   n_jobs=n_jobs)

                start = time.time()
                estimator.fit(X_train, y_train)
                fit_time = time.time() - start

                y_pred = estimator.predict(X_test)
                f1_macro = f1_score(y_test, y_pred, average='macro')
                f1_weighted = f1_score(y_test, y_pred, average='weighted')
                accuracy = (y_pred == y_test).mean()

                row = dict(n_estimators=int(n_estimators), max_depth=int(max_depth),
                           leaf_percent=float(leaf_percent), split_percent=float(split_percent),
                           min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                           category_size=category_size, f1_macro=f1_macro,
                           f1_weighted=f1_weighted, accuracy=accuracy, fit_time=fit_time)

                # Append immediately (resumable output)
                pd.DataFrame([row], columns=columns).to_csv(grid_file, mode='a',
                                                            header=False, index=False)

                print(f'{i_fit}/{n_total}) trees {n_estimators}, depth {max_depth}, '
                      f'leaf {leaf_percent}% ({min_samples_leaf}), '
                      f'split {split_percent}% ({min_samples_split}) '
                      f'-> F1 = {f1_macro:0.4f} ({fit_time:0.1f}s)')

print(f'\nScan completed in {(time.time() - start_scan) / 60:0.1f} minutes')
print(f'Output file: {grid_file}')