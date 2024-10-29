from joblib import load as jload
import numpy as np

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Specify the files location
_ASPECT_FOLDER = Path(__file__).parent
_MODEL_FOLDER = _ASPECT_FOLDER/'models'

# Configuration file
_CONF_FILE = _ASPECT_FOLDER/'aspect.toml'

class Aspect_Error(Exception):
    """Aspect exception function"""

# Read lime configuration file
with open(_CONF_FILE, mode="rb") as fp:
    cfg_aspect = tomllib.load(fp)

# Default feature detection model
DEFAULT_MODEL_ADDRESS = _MODEL_FOLDER/'training_multi_sample_v4_min-max_8categories_v4_175000points_angleSample_numpy_array_model.joblib'


def read_trained_model(file_address):

    # Read trained model
    model = jload(file_address)

    # Read lime configuration file
    cfg_address = Path(file_address).parent/f'{file_address.stem}.toml'
    with open(cfg_address, mode="rb") as cm:
        cfg_model = tomllib.load(cm)

    return model, cfg_model


def check_lisa(model1D, model2D, setup_cfg):

    if model1D is None:
        coeffs1D = np.array(setup_cfg['linear']['model1D_coeffs']), np.array(setup_cfg['linear']['model1D_intercept'])
    else:
        model1D_job = jload(model1D)
        coeffs1D = np.squeeze(model1D_job.coef_), np.squeeze(model1D_job.intercept_)

    if model2D is None:
        coeffs2D = np.array(setup_cfg['linear']['model2D_coeffs']), np.array(setup_cfg['linear']['model2D_intercept'])
    else:
        model2D_job = jload(model2D)
        coeffs2D = np.squeeze(model2D_job.coef_), np.squeeze(model2D_job.intercept_)

    return coeffs1D, coeffs2D