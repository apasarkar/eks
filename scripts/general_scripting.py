import os
import argparse

# -------------------------------------------------------------
""" Collection of General Functions for EKS Scripting
The functions here are called by individual example scripts """
# -------------------------------------------------------------


# ---------------------------------------------
# Command Line Arguments and File I/O
# ---------------------------------------------

# Finds + returns save directory if specified, otherwise defaults to outputs
def handle_io(input_dir, save_dir):
    if not os.path.isdir(input_dir):
        raise ValueError('--input-dir must be a valid directory containing prediction files')
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def handle_parse_args(script_type):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        required=True,
        help='directory of model prediction csv files',
        type=str,
    )
    parser.add_argument(
        '--save-dir',
        help='save directory for outputs (default is input-dir)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--save-filename',
        help='filename for outputs (default uses smoother type and s parameter)',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--data-type',
        help='format of input data (Lightning Pose = lp, SLEAP = slp), dlc by default.',
        default='dlc',
        type=str,
    )
    if script_type == 'singlecam':
        add_bodyparts(parser)
        add_s(parser)
    elif script_type == 'multicam':
        add_bodyparts(parser)
        add_camera_names(parser)
        add_quantile_keep_pca(parser)
        add_s(parser)
    elif script_type == 'pupil':
        add_diameter_s(parser)
        add_com_s(parser)
    elif script_type == 'paw':
        add_s(parser)
        add_quantile_keep_pca(parser)
    else:
        raise ValueError("Unrecognized script type.")
    args = parser.parse_args()
    return args


# Helper Functions for handle_parse_args
def add_bodyparts(parser):
    parser.add_argument(
        '--bodypart-list',
        required=True,
        nargs='+',
        help='the list of body parts to be ensembled and smoothed',
    )
    return parser


def add_s(parser):
    parser.add_argument(
        '--s',
        help='smoothing parameter ranges from .01-20 (smaller values = more smoothing)',
        type=float,
    )
    return parser


def add_camera_names(parser):
    parser.add_argument(
        '--camera-names',
        required=True,
        nargs='+',
        help='the camera names',
    )
    return parser


def add_quantile_keep_pca(parser):
    parser.add_argument(
        '--quantile_keep_pca',
        help='percentage of the points are kept for multi-view PCA (lowest ensemble variance)',
        default=25,
        type=float,
    )
    return parser


def add_diameter_s(parser):
    parser.add_argument(
        '--diameter-s',
        help='smoothing parameter for diameter (closer to 1 = more smoothing)',
        default=.9999,
        type=float
    )
    return parser


def add_com_s(parser):
    parser.add_argument(
        '--com-s',
        help='smoothing parameter for center of mass (closer to 1 = more smoothing)',
        default=.999,
        type=float
    )
    return parser
