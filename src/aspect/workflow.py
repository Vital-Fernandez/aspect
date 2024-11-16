import numpy as np
from time import time
from .io import read_trained_model, DEFAULT_MODEL_ADDRESS, cfg, Aspect_Error
from .tools import monte_carlo_expansion, feature_scaling, white_noise_scale
from matplotlib import pyplot as plt

CHOICE_DM = np.array(cfg['decision_matrices']['choice'])
TIME_DM = np.array(cfg['decision_matrices']['time'])

# TODO Larger box overwrites small
# TODO COMPLEX overwrites simple

def unpack_spec_flux(spectrum):

    pixels_mask = spectrum.flux.mask if np.ma.isMaskedArray(spectrum.flux) else np.zeros(spectrum.flux).astype(bool)
    flux_arr = spectrum.flux if not np.ma.isMaskedArray(spectrum.flux) else spectrum.flux.data[~spectrum.flux.mask]
    err_arr = spectrum.err_flux if not np.ma.isMaskedArray(spectrum.err_flux) else spectrum.err_flux.data[~spectrum.err_flux.mask]

    return flux_arr, err_arr, pixels_mask


def enbox_spectrum(input_flux, box_size, range_box):

    # Use only the true entries from the mask
    # flux_array = input_flux if not np.ma.isMaskedArray(input_flux) else input_flux.data[~input_flux.mask]

    # Reshape to the detection interval
    n_intervals = input_flux.size - box_size + 1
    input_flux = input_flux[np.arange(n_intervals)[:, None] + range_box]

    # # Remove nan entries
    # idcs_nan_rows = np.isnan(input_flux).any(axis=1)
    # flux_array = input_flux[~idcs_nan_rows, :]

    return input_flux


class ModelManager:

    def __init__(self, model_address=None, n_jobs=None, verbose=0):

        self.cfg = None
        self.detection_model = None
        self.size_arr = None
        self.scale = None
        self.log_base = None

        self.categories_str = None
        self.feature_number_dict = None
        self.number_feature_dict = None
        self.n_categories = None

        # Default values
        model_address = DEFAULT_MODEL_ADDRESS if model_address is None else model_address

        # Load the model
        self.predictor, self.cfg = read_trained_model(model_address)

        # Specify cores (default 4)
        n_jobs = 4
        self.predictor.n_jobs = n_jobs  # Use 4 cores
        self.predictor.verbose = verbose  # No output message

        # Array with the boxes size
        self.size_arr = np.atleast_1d(self.cfg['properties']['box_size'])

        # Scaling properties
        self.scale = self.cfg['properties']['scale']
        self.log_base = self.cfg['properties'].get('log_base')
        self.categories_str = np.array(self.cfg['properties']['categories'])
        self.feature_number_dict = cfg['shape_number']
        self.number_feature_dict = {v: k for k, v in self.feature_number_dict.items()}

        self.n_categories = len(self.feature_number_dict)

        return

    def reload_model(self, model_address=None, n_jobs=None):

        # Call the constructor again
        self.__init__(model_address, n_jobs)

        return


# Create object with default model
model = ModelManager()


class SpectrumDetector:

    def __init__(self, spectrum, model_address=None):

        self._spec = spectrum
        self.narrow_detect = None
        self.box_width = None
        self.range_box = None
        self.n_mc = 100
        self.detection_min = 40
        self.white_noise_maximum = 50

        self.line_1d_pred = None
        self.line_2d_pred = None
        self.line_pred = None

        self.features = None

        # Read the detection model
        if model_address is None:
            self.model = model

        # Arrays to store the data
        self.seg_flux = None
        self.seg_err = None

        self.seg_pred = None
        self.seg_conf = None

        self.pred_arr = None
        self.conf_arr = None

        return

    def detection(self, feature_list=None, bands=None, show_steps=False):

        # Support variables
        n_pixels = self._spec.flux.size
        box_size = self.model.size_arr[0]
        box_range = np.arange(box_size)

        # Empty containers
        self.pred_arr = np.zeros(n_pixels, dtype=np.int64)
        self.conf_arr = np.zeros(n_pixels, dtype=np.int64)
        self.seg_pred = np.zeros(box_size, dtype=np.int64)
        self.seg_conf = np.zeros(box_size, dtype=np.int64)

        # Remove masks from flux and uncertainty
        y_arr, err_arr, pixel_mask = unpack_spec_flux(self._spec)
        # invers_mask = ~pixel_mask

        if (y_arr.size > box_size) or ~np.all(pixel_mask):

            # Reshape spectrum to box size
            y_enbox = enbox_spectrum(y_arr, box_size, box_range)
            err_enbox = enbox_spectrum(err_arr, box_size, box_range)

            # MC expansion
            y_enbox =  monte_carlo_expansion(y_enbox, err_enbox, self.n_mc, for_loop=False)

            # Scaling
            y_norm = feature_scaling(y_enbox, 'min-max', 1)

            # Run the prediction
            y_reshaped = y_norm.transpose(0, 2, 1).reshape(-1, box_size)
            y_pred = self.model.predictor.predict(y_reshaped)
            y_pred = y_pred.reshape(-1, 100)

            # Get the count of types detected on Monte-Carlo
            counts_categories = np.apply_along_axis(np.bincount, axis=1, arr=y_pred, minlength=self.model.n_categories)

            # Loop through non white-noise regions:
            idcs_valid = np.nonzero(counts_categories[:, 1] < self.white_noise_maximum)[0]
            for idx in idcs_valid:

                # Get segment arrays
                self.seg_pred[:] = self.pred_arr[idx:idx + box_size]
                self.seg_conf[:] = self.conf_arr[idx:idx + box_size]

                # Count
                counts = counts_categories[idx, :]
                idcs_categories = counts > self.detection_min

                # Choose detection
                out_type, out_confidence = self.detection_evaluation(counts, idcs_categories)

                # Check with previous detection
                idcs_pred, new_pred, new_conf = self.detection_revision(idx, box_size, out_type, out_confidence)

                # Only pass if more than half
                half_check = idcs_pred[6:].sum() > 5
                if half_check:
                    idcs_pred = np.nonzero(idcs_pred)
                    self.seg_pred[idcs_pred] = new_pred[idcs_pred]
                    self.seg_conf[idcs_pred] = new_conf[idcs_pred]
                else:
                    self.seg_pred[:] = self.pred_arr[idx:idx + box_size]
                    self.seg_conf[:] = self.conf_arr[idx:idx + box_size]

                if show_steps:
                    self.plot_steps(y_norm[idx, :], idx, counts, idcs_categories, out_type, out_confidence,
                                    self.pred_arr[idx:idx + box_size], self.conf_arr[idx:idx + box_size],
                                    idcs_pred, new_pred, new_conf)

                # Assign new categories and confidence
                self.pred_arr[idx:idx + box_size] = self.seg_pred[:]
                self.conf_arr[idx:idx + box_size] = self.seg_conf[:]



    def detection_evaluation(self, counts_categories, idcs_categories):

        n_detections = idcs_categories.sum()

        match n_detections:

            # Undefined
            case 0:
                return 0, 0

            # One detection
            case 1:
                return np.argmax(idcs_categories), counts_categories[idcs_categories][0]

            # Two detections
            case 2:
                category_candidates = np.nonzero(idcs_categories)[0]
                idx_output = CHOICE_DM[category_candidates[0], category_candidates[1]]
                output_type, output_count = category_candidates[idx_output], counts_categories[idcs_categories][idx_output]
                return output_type, output_count

            # Three detections
            case _:
                raise Aspect_Error(f'Number of detections: "{n_detections}" is not recognized')


    def detection_revision(self, idx, box_size, new_type, new_confidence):

        new_pred, new_conf = np.full(box_size, new_type), np.full(box_size, new_confidence)
        idcs_pred = TIME_DM[self.seg_pred, new_pred]
        # idcs_pred = np.nonzero(idcs_pred)

        return idcs_pred, new_pred, new_conf


    def transform_category(self, input_category, segment_flux):

        match input_category:

            # White noise scale
            case 1:
                return white_noise_scale(segment_flux)

            case _:
                return input_category

    def plot_steps(self, y_norm, idx, counts, idcs_categories, out_type, out_confidence, old_pred, old_conf,
                   idcs_pred, new_pred, new_conf):

        x_arr = self._spec.wave_rest if not np.ma.isMaskedArray(self._spec.wave_rest) else self._spec.wave_rest.data[~self._spec.wave_rest.mask]
        x_sect = x_arr[idx:idx+y_norm.shape[0]]
        print(f'Idx "{idx}"; counts: {counts}; Output: {model.number_feature_dict[out_type]} ({out_type})')

        colors_old = [cfg['colors'][model.number_feature_dict[val]] for val in old_pred]
        colors_new = [cfg['colors'][model.number_feature_dict[val]] for val in self.seg_pred]

        fig, ax = plt.subplots()
        color_detection = cfg['colors'][model.number_feature_dict[out_type]]
        ax.step(x_sect, y_norm[:,0], where='mid', color=color_detection, label='Out detection')
        ax.scatter(x_sect, np.zeros(x_sect.size), color=colors_old, label='Old prediction')
        ax.scatter(x_sect, np.ones(x_sect.size), color=colors_new, label='New prediction')
        ax.set_xlabel(r'Wavelength $(\AA)$')

        ax_secondary = ax.twinx()  # Creates a twin y-axis on the right
        ax_secondary.set_ylim(ax.get_ylim())  # Match the primary y-axis limits
        ax_secondary.set_yticks([0, 0.5, 1])  # Custom tick positions
        ax_secondary.set_yticklabels(['Previous\nClassification', 'Present\nClassification', 'Output\nClassification'])

        plt.tight_layout()
        plt.show()

        # fig, ax = plt.subplots()
        # ax.step(np.arange(self.seg_flux.size), self.seg_flux, color=cfg_aspect['colors'][aspect_model.number_feature_dict[output_type]])
        # ax.set_title(f'{aspect_model.number_feature_dict[output_type]} ({output_type})')
        # plt.show()


        return