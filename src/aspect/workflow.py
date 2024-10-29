import numpy as np
from time import time
from .io import read_trained_model, DEFAULT_MODEL_ADDRESS, cfg_aspect, Aspect_Error
from .tools import monte_carlo_expansion, feature_scaling, white_noise_scale
from matplotlib import pyplot as plt

CHOICE_DM = np.array(cfg_aspect['decision_matrices']['choice'])
TIME_DM = np.array(cfg_aspect['decision_matrices']['time'])


def unpack_spec_flux(spectrum):

    flux_arr = spectrum.flux if not np.ma.isMaskedArray(spectrum.flux) else spectrum.flux.data[~spectrum.flux.mask]
    err_arr = spectrum.err_flux if not np.ma.isMaskedArray(spectrum.err_flux) else spectrum.err_flux.data[~spectrum.err_flux.mask]

    return flux_arr, err_arr


def enbox_spectrum(input_flux, box_size, range_box):

    # Use only the true entries from the mask
    flux_array = input_flux if not np.ma.isMaskedArray(input_flux) else input_flux.data[~input_flux.mask]

    # Reshape to the detection interval
    n_intervals = flux_array.size - box_size + 1
    flux_array = flux_array[np.arange(n_intervals)[:, None] + range_box]

    # # Remove nan entries
    # idcs_nan_rows = np.isnan(input_flux).any(axis=1)
    # flux_array = input_flux[~idcs_nan_rows, :]

    return flux_array


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
        self.feature_number_dict = self.cfg['features_number']
        self.number_feature_dict = {v: k for k, v in self.feature_number_dict.items()}

        self.n_categories = len(self.feature_number_dict)

        return

    def reload_model(self, model_address=None, n_jobs=None):

        # Call the constructor again
        self.__init__(model_address, n_jobs)

        return


# Create object with default model
aspect_model = ModelManager()


class SpectrumDetector:

    def __init__(self, spectrum, model_address=None):

        self._spec = spectrum
        self.narrow_detect = None
        self.box_width = None
        self.range_box = None
        self.n_mc = 100
        self.detection_min = 40

        self.line_1d_pred = None
        self.line_2d_pred = None
        self.line_pred = None

        self.features = None

        # Read the detection model
        if model_address is None:
            self.model = aspect_model

        # Arrays to store the data
        self.seg_flux = None
        self.seg_err = None

        self.seg_pred = None
        self.conf_pred = None

        self.pred_arr = None
        self.conf_arr = None

        return

    def detection(self, feature_list=None, bands=None):

        # Empty container for the data
        n_pixels = self._spec.flux.size
        self.pred_arr = np.zeros(n_pixels).astype(int)
        self.conf_arr = np.zeros(n_pixels).astype(int)

        y_arr, err_arr = unpack_spec_flux(self._spec)

        # Loop through the pixels and box sizes
        for idx in np.arange(n_pixels):
            for box_size in self.model.size_arr:
                if idx < (n_pixels - box_size):

                    # Box flux and uncertainty
                    self.seg_flux = y_arr[idx:idx+box_size]
                    self.seg_err = err_arr[idx:idx+box_size]

                    # Monte-carlo
                    seg_matrix = monte_carlo_expansion(self.seg_flux, self.seg_err, self.n_mc)

                    # Normalization
                    seg_matrix = feature_scaling(seg_matrix, self.model.scale, self.model.log_base)

                    # Feature detection
                    start_time = time()
                    seg_string_pred = self.model.predictor.predict(seg_matrix)
                    fit_time = np.round((time() - start_time), 5)
                    print(f'- MC ({fit_time} seconds)')

                    # coso = np.tile(seg_matrix, (n_pixels, 1))
                    # start_time = time()
                    # coso_pred = self.model.predictor.predict(coso)
                    # fit_time = np.round((time() - start_time), 5)
                    # print(f'- MC ({fit_time} seconds)')

                    # Convert string array to integer array using vectorized approach
                    # self.seg_pred = np.vectorize(self.model.feature_number_dict.get)(seg_string_pred)
                    self.seg_pred = seg_string_pred

                    # Get the type and number of detections
                    counts_categories = np.bincount(self.seg_pred, minlength=self.model.n_categories)
                    idcs_categories = counts_categories > self.detection_min

                    # Decide between categories
                    output_type, output_confidence = self.detection_evaluation(counts_categories, idcs_categories)

                    # print(f'count_categories: {counts_categories}')
                    # print(f'output_type:{aspect_model.number_feature_dict[output_type]} ({output_type}), confidence: {output_confidence}, ')


                    # Transform categories
                    output_type = self.transform_category(output_type, self.seg_flux)

                    # print(f'Transform: {aspect_model.number_feature_dict[output_type]} ({output_type})')

                    #
                    # fig, ax = plt.subplots()
                    # ax.step(np.arange(self.seg_flux.size), self.seg_flux, color=cfg_aspect['colors'][aspect_model.number_feature_dict[output_type]])
                    # ax.set_title(f'{aspect_model.number_feature_dict[output_type]} ({output_type})')
                    # plt.show()


                    # Check with previous detection
                    new_pred, new_conf = np.full(box_size, output_type), np.full(box_size, output_confidence)
                    idcs_output_type  = TIME_DM[self.pred_arr[idx:idx+box_size], new_pred]
                    # output_confidence  = TIME_DM[self.conf_arr[idx:idx+box_size], new_pred]

                    # Assign values to array
                    if output_type != 0:
                        self.pred_arr[idx:idx+box_size][idcs_output_type] = output_type
                        self.conf_arr[idx:idx+box_size][idcs_output_type] = output_confidence

    def detection_loopless(self, feature_list=None, bands=None):

        # Empty container for the data
        n_pixels = self._spec.flux.size
        self.pred_arr = np.zeros(n_pixels).astype(int)
        self.conf_arr = np.zeros(n_pixels).astype(int)

        y_arr, err_arr = unpack_spec_flux(self._spec)

        # Reshape spectrum to box size
        box_size = self.model.size_arr[0]
        box_range = np.arange(box_size)
        y_enbox = enbox_spectrum(y_arr, self.model.size_arr[0], box_range)
        err_enbox = enbox_spectrum(err_arr, self.model.size_arr[0], box_range)

        # MC expansion
        y_enbox =  monte_carlo_expansion(y_enbox, err_enbox, self.n_mc, for_loop=False)

        # Scaling
        y_norm = feature_scaling(y_enbox, 'min-max', 1)

        # Run the prediction
        y_reshaped = y_norm.transpose(0, 2, 1).reshape(-1, box_size)
        y_pred = self.model.predictor.predict(y_reshaped)
        y_pred = y_pred.reshape(-1, 100)

        counts_categories = np.apply_along_axis(np.bincount, 1, y_pred, minlength=self.model.n_categories)
        idcs_categories = counts_categories > self.detection_min



    # # Reshape (X, 12, 100) -> (X * 100, 12)
# X_mc_reshaped = X_mc.transpose(0, 2, 1).reshape(-1, 12)
#
# # Apply the random forest model to get predictions
# predictions = model.predict(X_mc_reshaped)
#
# # Reshape the predictions (X * 100,) -> (X, 100)
# predictions_reshaped = predictions.reshape(-1, 100)

        # # Loop through the pixels and box sizes
        # for idx in np.arange(n_pixels):
        #     for box_size in self.model.size_arr:
        #         if idx < (n_pixels - box_size):
        #
        #             # Box flux and uncertainty
        #             self.seg_flux = y_arr[idx:idx+box_size]
        #             self.seg_err = err_arr[idx:idx+box_size]
        #
        #             # Monte-carlo
        #             seg_matrix = monte_carlo_expansion(self.seg_flux, self.seg_err, self.n_mc)
        #
        #             # Normalization
        #             seg_matrix = feature_scaling(seg_matrix, self.model.scale, self.model.log_base)
        #
        #             # Feature detection
        #             seg_string_pred = self.model.predictor.predict(seg_matrix)
        #
        #             coso = np.tile(seg_matrix, (n_pixels, 1))
        #             start_time = time()
        #             coso_pred = self.model.predictor.predict(coso)
        #             fit_time = np.round((time() - start_time), 5)
        #             print(f'- MC ({fit_time} seconds)')
        #
        #             # Convert string array to integer array using vectorized approach
        #             # self.seg_pred = np.vectorize(self.model.feature_number_dict.get)(seg_string_pred)
        #             self.seg_pred = seg_string_pred
        #
        #             # Get the type and number of detections
        #             counts_categories = np.bincount(self.seg_pred, minlength=self.model.n_categories)
        #             idcs_categories = counts_categories > self.detection_min
        #
        #
        #             # Decide between categories
        #             output_type, output_confidence = self.detection_evaluation(counts_categories, idcs_categories)
        #
        #             # print(f'count_categories: {counts_categories}')
        #             # print(f'output_type:{aspect_model.number_feature_dict[output_type]} ({output_type}), confidence: {output_confidence}, ')
        #
        #
        #             # Transform categories
        #             output_type = self.transform_category(output_type, self.seg_flux)
        #
        #             # print(f'Transform: {aspect_model.number_feature_dict[output_type]} ({output_type})')
        #
        #             #
        #             # fig, ax = plt.subplots()
        #             # ax.step(np.arange(self.seg_flux.size), self.seg_flux, color=cfg_aspect['colors'][aspect_model.number_feature_dict[output_type]])
        #             # ax.set_title(f'{aspect_model.number_feature_dict[output_type]} ({output_type})')
        #             # plt.show()
        #
        #
        #             # Check with previous detection
        #             new_pred, new_conf = np.full(box_size, output_type), np.full(box_size, output_confidence)
        #             idcs_output_type  = TIME_DM[self.pred_arr[idx:idx+box_size], new_pred]
        #             # output_confidence  = TIME_DM[self.conf_arr[idx:idx+box_size], new_pred]
        #
        #             # Assign values to array
        #             if output_type != 0:
        #                 self.pred_arr[idx:idx+box_size][idcs_output_type] = output_type
        #                 self.conf_arr[idx:idx+box_size][idcs_output_type] = output_confidence


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
                category_candidates = np.where(idcs_categories)[0]
                print(f'Double categories: {aspect_model.number_feature_dict[category_candidates[0]]} {aspect_model.number_feature_dict[category_candidates[1]]} ')
                idx_output = CHOICE_DM[category_candidates[0], category_candidates[1]]
                output_type, output_count = category_candidates[idx_output], counts_categories[idcs_categories][idx_output]
                print(f'- output: {aspect_model.number_feature_dict[output_type]}')
                return output_type, output_count

            # Three detections
            case _:
                raise Aspect_Error(f'Number of detections: "{n_detections}" is not recognized')


    def transform_category(self, input_category, segment_flux):

        match input_category:

            # White noise scale
            case 1:
                return white_noise_scale(segment_flux)

            case _:
                return input_category