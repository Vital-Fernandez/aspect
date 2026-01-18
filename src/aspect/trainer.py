import importlib
import numpy as np
import joblib
import toml
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from time import time
from pathlib import Path
from .io import cfg as aspect_cfg


def get_training_test_sets(x_arr, y_arr, test_fraction, n_pixel_features, n_scale_features, random_state=None, classification=True):

    # Split into training and testing:


    if classification:

        print(f'\nSplitting sample with categories:')
        print(np.unique(y_arr))
        sss = StratifiedShuffleSplit(n_splits=1, train_size=int(y_arr.size * (1 - test_fraction)),
                                     test_size=int(y_arr.size * test_fraction), random_state=random_state)

        # Equal splits
        for train_index, test_index in sss.split(x_arr, y_arr):
            X_train, X_test = x_arr[train_index, :], x_arr[test_index, :]
            y_train, y_test = y_arr[train_index], y_arr[test_index]

        # Convert strings to integers
        y_train = np.vectorize(aspect_cfg['shape_number'].get)(y_train)
        y_test = np.vectorize(aspect_cfg['shape_number'].get)(y_test)

    else:
        # feature_slice = -n_pixel_features - n_scale_features
        # X_train, X_test, y_train, y_test = train_test_split(x_arr[:, feature_slice:],
        #                                                     y_arr,
        #                                                     test_size=test_fraction,
        #                                                     random_state=random_state, shuffle=True)

        X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=test_fraction,
                                                            random_state=random_state, shuffle=True)


    return X_train, y_train, X_test, y_test


def components_trainer(model_label, x_arr, y_arr, fit_cfg, list_labels, output_folder=None, test_fraction=0.1,
                       random_state=None, classification=True):

    # Preparing the estimator:
    print(f'\nLoading estimator: {fit_cfg["estimator"]["class"]}')
    estimator = getattr(importlib.import_module(fit_cfg['estimator']["module"]), fit_cfg['estimator']["class"])
    estimator_params = fit_cfg.get('estimator_params', {})

    # Split into training and testing:
    data_train, y_train, data_test, y_test = get_training_test_sets(x_arr, y_arr, test_fraction,
                                                              n_pixel_features=fit_cfg['box_size'], n_scale_features=1,
                                                              random_state=random_state, classification=classification)

    # Select just the features
    feature_slice = -fit_cfg['box_size'] - 1
    X_train, X_test = data_train[:, feature_slice:], data_test[:, feature_slice:]

    # Run the training
    if classification:
        print(f'\nClassification: {y_train.size/len(fit_cfg["categories"]):.0f} * {len(fit_cfg["categories"])} = {y_train.size}  points ({model_label})')
        print(f'- Settings: {fit_cfg["estimator_params"]}\n')
    else:
        print(f'Regression range: [{y_train.min():.3f}, {y_train.max():.3f}]')

    start_time = time()
    ml_function = estimator(**estimator_params)
    ml_function.fit(X_train, y_train)
    end_time = np.round((time()-start_time)/60, 2)
    print(f'- completed ({end_time} minutes)')

    # Save the trained model and configuration
    output_folder = Path(output_folder)/'results'
    output_folder.mkdir(parents=True, exist_ok=True)

    model_address = output_folder/f'{model_label}.joblib'
    joblib.dump(ml_function, model_address)

    if classification:

        # Run initial diagnostics
        print(f'\nReloading model from: {model_address}')
        start_time = time()
        ml_function = joblib.load(model_address)
        fit_time = np.round((time()-start_time), 3)
        print(f'- completed ({fit_time} seconds)')

        print(f'\nRuning prediction on test set ({y_test.size} points)')
        start_time = time()
        y_pred = ml_function.predict(X_test)
        print(f'- completed ({(time()-start_time):0.1f} seconds)')

        # Testing confussion matrix
        print(f'\nConfusion matrix in test set ({y_test.size} points)')
        start_time = time()
        conf_matrix_test = confusion_matrix(y_test, y_pred, normalize="all")
        print(f'- completed ({(time()-start_time):0.1f} seconds)')

        # Precision, recall and f1:
        print(f'\nF1, Precision and recall diagnostics ({y_test.size} points)')
        start_time = time()
        pres = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f'- completed ({(time()-start_time):0.1f} seconds)')

        print(f'\nModel outputs')
        print(f'- F1: \n {f1}')
        print(f'- Precision: \n {pres}')
        print(f'- Recall: \n {recall}')
        print(f'- Testing confusion matrix: \n {conf_matrix_test}')
        print(f'- Fitting time (seconds): \n {float(fit_time)}')

        # Save results into a TOML file
        toml_path = output_folder/f'{model_label}.toml'
        output_dict = {'resuts': {'f1':f1, 'precision':pres, 'Recall':recall, 'confusion_matrix':conf_matrix_test,
                                  'fit_time': fit_time}, 'properties': fit_cfg,}
        with open(toml_path, 'w') as f:
            toml.dump(output_dict, f)

    else:

        # Reload model
        print(f'\nReloading model from: {model_address}')
        start_time = time()
        ml_function = joblib.load(model_address)
        fit_time = np.round((time() - start_time), 3)
        print(f'- completed ({fit_time} seconds)')

        # Prediction
        print(f'\nRunning prediction on test set ({y_test.size} points)')
        start_time = time()
        y_pred = ml_function.predict(X_test)
        pred_time = np.round((time() - start_time), 3)
        print(f'- completed ({pred_time} seconds)')

        # Core regression metrics
        print(f'\nRegression diagnostics ({y_test.size} points)')
        start_time = time()

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Normalized errors (scale-independent)
        y_range = y_test.max() - y_test.min()
        nrmse = rmse / y_range if y_range > 0 else np.nan
        nmae = mae / y_range if y_range > 0 else np.nan

        print(f'- completed ({(time() - start_time):0.1f} seconds)')

        # Outputs
        print(f'\nModel outputs')
        print(f'- R²: \n {r2}')
        print(f'- RMSE: \n {rmse}')
        print(f'- MAE: \n {mae}')
        print(f'- Median AE: \n {medae}')
        print(f'- Normalized RMSE: \n {nrmse}')
        print(f'- Normalized MAE: \n {nmae}')
        print(f'- Fit time (seconds): \n {float(fit_time)}')

        # Save results to TOML
        toml_path = output_folder / f'{model_label}.toml'
        output_dict = {
            'results': {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae),
                'median_ae': float(medae),
                'nrmse': float(nrmse),
                'nmae': float(nmae),
                'fit_time': float(fit_time),
                'prediction_time': float(pred_time),
            },
            'properties': fit_cfg,
        }

        # Scatter plot
        fig, ax = plt.subplots()

        idcs_limit = 5000
        ycoords, xcoords = data_test[:, 0], data_test[:, 1]
        error = y_test - y_pred  # signed error
        abs_error = np.abs(error)
        rel_error = error / y_test
        limit = np.percentile(rel_error, 95)

        # Set the color limits

        sc = ax.scatter(xcoords[:idcs_limit], ycoords[:idcs_limit], c=rel_error[:idcs_limit], s=8, cmap='viridis')
        sc.set_clim(-limit, limit)

        cbar = fig.colorbar(sc, ax=ax, label='|Prediction error|')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.show()

        with open(toml_path, 'w') as f:
            toml.dump(output_dict, f)


    return
