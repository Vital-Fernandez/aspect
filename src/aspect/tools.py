import logging
import numpy as np
from .io import Aspect_Error

# Log variable
_logger = logging.getLogger('aspect')


def monte_carlo_expansion(flux_array, err_array, n_mc, for_loop=True):


    # # Get the noise scale for the selections
    # if noise_scale is None:
    #     noise_scale = self._spec.err_flux if self._spec.err_flux is not None else self._spec.cont_std
    #
    #     if noise_scale is None:
    #         _logger.warning(f"No flux uncertainty provided for the line detection. There won't be a confidence value"
    #                         f" for the predictions.")
    #         self.n_mc = 1

    # Scale array depending on the scale
    # noise_scale = err_array if np.isscalar(err_array) else err_array[..., None]
    # noise_scale = err_array

    # Add random noise matrix
    # noise_array = np.random.normal(0, err_array, size=(n_mc, flux_array.size))
    # mc_flux = flux_array[:, :, np.newaxis] + noise_array

    if for_loop:
        mc_flux = flux_array + np.random.normal(0, err_array, size=(n_mc, flux_array.size))

    else:
        noise_scale = err_array if np.isscalar(err_array) else err_array[..., None]
        noise_matrix_shape = (flux_array.shape[0], flux_array.shape[1], n_mc)

        noise_array = np.random.normal(0, noise_scale, size=noise_matrix_shape)
        mc_flux = flux_array[:, :, np.newaxis] + noise_array

    return mc_flux


def scale_min_max(data, axis=None):

    data_min_array = data.min(axis=axis, keepdims=True)
    data_max_array = data.max(axis=axis, keepdims=True)
    data_norm = (data - data_min_array) / (data_max_array - data_min_array)

    return data_norm


def scale_log(data, log_base, axis=None):

    data_min_array = data.min(axis=axis, keepdims=True)

    y_cont = data - data_min_array + 1
    data_norm = np.emath.logn(log_base, y_cont)

    return data_norm


def scale_log_min_max(data, log_base, axis=None):

    data_min_array = data.min(axis=axis, keepdims=True)
    data_cont = data - data_min_array + 1
    log_data = np.emath.logn(log_base, data_cont)
    log_min_array, log_max_array = log_data.min(axis=axis, keepdims=True), log_data.max(axis=axis, keepdims=True)
    data_norm = (log_data - log_min_array) / (log_max_array - log_min_array)

    return data_norm


def feature_scaling(data, transformation, log_base=None, axis=1):

    match transformation:
        case 'min-max':
            return scale_min_max(data, axis=axis)
        case 'log':
            return scale_log(data, log_base=log_base, axis=axis)
        case 'log-min-max':
            return scale_log_min_max(data, log_base=log_base, axis=axis)
        case _:
            raise Aspect_Error(f'Input scaling: "{transformation}" is not recognized')


def white_noise_scale(flux_arr):

    min, max = flux_arr.min(), flux_arr.max()

    diff = max - min if max != 0 else np.abs(max-min)

                # 1 White noise, 2 continuum
    output_type = 1 if diff > 10 else 2

    return output_type