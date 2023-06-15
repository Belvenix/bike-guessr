import os
from pathlib import Path

# Download graph env variables
GRAPHML_TRAIN_DATA_DIR = Path(os.getenv('GRAPHML_TRAIN_DATA_DIR', './src/docker_data/data/data_train'))
GRAPHML_VALIDATION_DATA_DIR = Path(os.getenv('GRAPHML_VALIDATION_DATA_DIR', './src/docker_data/data/data_val'))
GRAPHML_TEST_DATA_DIR = Path(os.getenv('GRAPHML_TEST_DATA_DIR', './src/docker_data/data/data_test'))
GRAPHML_TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
GRAPHML_VALIDATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
GRAPHML_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Transform graph env variables
TRANSFORM_DATA_OUTPUT_DIR = Path(os.getenv('TRANSFORM_DATA_OUTPUT_DIR', './src/docker_data/data/data_transformed'))
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
CLASSIFIER_TEST_DATA_PATH = Path(os.getenv('CLASSIFIER_TEST_DATA_PATH', './src/docker_data/data/data_transformed/test.bin'))
CLASSIFIER_WEIGHTS_SAVE_DIR = Path(os.getenv('CLASSIFIER_WEIGHTS_SAVE_DIR', './src/docker_data/data/weights'))
CLASSIFIER_OUTPUTS_SAVE_DIR = Path(os.getenv('CLASSIFIER_OUTPUTS_SAVE_DIR', './src/docker_data/data/outputs'))
PLOT_SAVE_DIR = Path(os.getenv('PLOT_SAVE_DIR', './src/docker_data/data/plots'))
CLASSIFIER_WEIGHTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFIER_OUTPUTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Visualize settings
VISUALIZATION_OUTPUT_DIR = Path(os.getenv('VISUALIZATION_OUTPUT_DIR', './src/docker_data/data/visualization'))
VISUALIZATION_LEGEND_PATH = Path(os.getenv('VISUALIZATION_LEGEND_PATH', './src/imgs/bicycle_prediction_legend.png'))
VISUALIZATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Download categories filter
# Cycle lanes in bidirectional motor car roads label=1
L_FILTER = [
    '["highway"]["cycleway"~"lane"]',
    '["highway"]["cycleway:left"="lane"]["cycleway:right"="lane"]',
    '["highway"]["cycleway:both"="lane"]',
    '["highway"]["cycleway:right"="lane"]["cycleway:right:oneway"!="yes"]',
]

# Cycle lanes in oneway motor car roads label=1
M_FILTER = [
    '["highway"]["oneway"="yes"]["cycleway:left"="lane"]["cycleway:right"="lane"]["oneway:bicycle"="no"]',
    '["highway"]["oneway"="yes"]["cycleway"="lane"]["oneway:bicycle"="no"]'

    '["highway"]["oneway"="yes"]["cycleway:right"="lane"]',
    '["highway"]["oneway"="yes"]["cycleway"="lane"]'

    '["highway"]["oneway"="yes"]["cycleway:left"="lane"]',
    '["highway"]["oneway"="yes"]["cycleway"="lane"]'

    '["highway"]["oneway"="yes"]["lanes"="2"]["cycleway"="lane"]'

    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway:left"="lane"]["cycleway:left:oneway"="no"]'

    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway:left"="lane"]["cycleway:left:oneway"="-1"]',
    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway"="opposite_lane"]'

    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway:right"="lane"]["cycleway:right:oneway"="-1"]',
    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway"="opposite_lane"]'
]

# Cycle tracks label=2
T_FILTER = [
    '["highway"]["bicycle"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="yes"]',
    '["highway"]["cycleway"~"track"]'

    '["highway"]["bicycle"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="no"]',
    '["highway"]["cycleway:right"~"track"]["cycleway:right:oneway"="no"]'

    '["highway"]["oneway"="yes"]["bicycle"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="no"]',
    '["highway"]["oneway"="yes"]["cycleway:right"~"track"]["oneway:bicycle"="no"]'

    '["highway"]["bicycle"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="yes"]'
]

# Miscellaneous label=2
S_FILTER = [
    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]',
    '["highway"]["oneway"="yes"]["oneway:bicycle"="no"]["cycleway"~"opposite"]'

    '["highway"]["cycleway:right"~"lane"]["bicycle:backward"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="yes"]',
    '["highway"]["cycleway:left"~"track"]["cycleway:right"~"lane"]'

    '["highway"]["cycleway"~"track"]["segregated"="yes"]',
    '["highway"]["bicycle"~"use_sidepath"]',
    '["highway"~"cycleway"]["oneway"="yes"]',
    '["highway"~"path"]["bicycle"~"designated"]["oneway"="yes"]["foot"~"designated"]["segregated"="yes"]',
    '["highway"~"cycleway"]["oneway"="yes"]["foot"~"designated"]["segregated"="yes"]',

    '["highway"]["cycleway"~"track"]["segregated"="yes"]["foot"~"designated"]',

    '["highway"~"path"]["segregated"="yes"]["foot"~"designated"]["bicycle"~"designated"]["surface"~"paved"]',

    '["ramp:bicycle"="yes"]',
    '["ramp:stroller"="yes"]',
    '["ramp:wheelchair"="yes"]',

    '["highway"~"(footway|cycleway|path)"]["bicycle"~"designated"]["foot"~"designated"]["segregated"="no"]["surface"]',

    '["highway"~"footway"]["bicycle"="yes"]'
]

# Cycle lanes and bus/taxi lanes label=1
B_FILTER = [
    '["highway"]["cycleway:right"="share_busway"]',
    '["highway"]["cycleway:left"="share_busway"]'
]

# Cycle streets and bicycle roads label=3
R_FILTER = [
    '["highway"]["cyclestreet"="yes"]',
    '["highway"]["bicycle_road"="yes"]'
]