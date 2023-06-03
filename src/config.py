import os
from pathlib import Path

# Download graph env variables
GRAPHML_TRAIN_DATA_DIR = Path(os.getenv('GRAPHML_TRAIN_DATA_DIR', './docker_data/data/data_train'))
GRAPHML_VALIDATION_DATA_DIR = Path(os.getenv('GRAPHML_VALIDATION_DATA_DIR', './docker_data/data/data_val'))
GRAPHML_TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
GRAPHML_VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Transform graph env variables
TRANSFORM_DATA_OUTPUT_DIR = Path(os.getenv('TRANSFORM_DATA_OUTPUT_DIR', './docker_data/data/data_transformed'))
TRANSFORM_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tensorboard log dir
TENSORBOARD_LOG_DIR = Path(os.getenv('TENSORBOARD_LOG_DIR', './src/docker_data/logs'))
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Encoder settings
ENCODER_WEIGHTS_PATH = Path(os.getenv('ENCODER_WEIGHTS_PATH', './src/encoder-weights.bin'))
ENCODER_CONFIG_PATH = Path(os.getenv('ENCODER_CONFIG_PATH', './src/encoder-config.yml'))

# Classifier settings
CLASSIFIER_TRAIN_DATA_PATH = Path(os.getenv('CLASSIFIER_TRAIN_DATA_PATH', './src/docker_data/data/data_transformed/train.bin'))
CLASSIFIER_VALIDATION_DATA_PATH = Path(os.getenv('CLASSIFIER_VALIDATION_DATA_PATH', './src/docker_data/data/data_transformed/validation.bin'))
CLASSIFIER_WEIGHTS_SAVE_DIR = Path(os.getenv('CLASSIFIER_WEIGHTS_SAVE_DIR', './src/docker_data/data/weights'))
CLASSIFIER_OUTPUTS_SAVE_DIR = Path(os.getenv('CLASSIFIER_OUTPUTS_SAVE_DIR', './src/docker_data/data/outputs'))
PLOT_SAVE_DIR = Path(os.getenv('PLOT_SAVE_DIR', './src/docker_data/data/plots'))
CLASSIFIER_WEIGHTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFIER_OUTPUTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
