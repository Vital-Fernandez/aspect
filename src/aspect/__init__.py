import logging
from .io import cfg, load_cfg, load_model
from .workflow import SpectrumDetector, model, CHOICE_DM, TIME_DM
from .plots import decision_matrix_plot
from .trainer import components_trainer

# Creating the lime logger
_logger = logging.getLogger("aspect")
_logger.setLevel(logging.INFO)

# Invert the dictionary of categories number
cfg['number_shape'] =  {v: k for k, v in cfg['shape_number'].items()}