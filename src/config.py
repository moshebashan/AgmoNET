# -*- coding: utf-8 -*-
"""
config.py

This module contains the configuration settings for the project, including paths,
data processing parameters, model training parameters, and other constants.

Environment-specific paths are set based on the WORKING_ENV variable, allowing for
flexibility when working across different environments (e.g., 'work', 'home', 'other').

Configuration settings are used throughout the project to maintain consistency
and make the codebase easier to manage. This module also reads the species mapping
file ('milon.txt') once, making the data available globally for other scripts.

Attributes:
    WORKING_ENV (str): Defines the working environment ('work', 'home', 'other').
    ROOT_PATH (str): The root path for the project based on the working environment.
    MILON_PATH (str): Path to the directory containing species mapping file 'milon.txt'.
    INPUT_DATA_PATH (str): Path to the directory containing input data recordings.
    OUTPUT_DATA_PATH_LOC (str): Path for saving output data.
    SPECIES_ACTIVITY_PATH (str): Path to the directory with species activity data.
    DF2_PATH (str): Path for additional data as required by the environment.

    SR (int): Sampling rate for audio processing.
    HOP_LENGTH (int): Hop length for spectrogram computation.
    N_FFT (int): Number of FFT components for spectrogram.
    MIN_FREQ (int): Minimum frequency for spectrogram.
    MAX_FREQ (int): Maximum frequency for spectrogram.
    MEL (bool): Whether to use Mel scale for spectrogram.
    N_MELS (int): Number of Mel bands if MEL is True.
    SEGMENT_DURATION (int): Duration of segments in seconds.
    OVERLAP (int): Overlap between audio segments in seconds.
    TRAIN_SIZE (float): Proportion of data used for training.

    N_AUG (int): Number of augmentations per signal for data augmentation.
    EPOCHS (int): Number of training epochs for models.
    BATCH_SIZE (int): Batch size for training models.
    VALIDATION_SPLIT (float): Ratio of data to use for validation during training.

    SPECIES_LIST (list): List of species IDs for activity analysis.
    STATIONS (list): List of station identifiers.
    DATE_START (str): Start date for analysis (format M-D-Y).
    DATE_END (str): End date for analysis (format M-D-Y).
    DST_TRANSITION (bool): Indicates whether to consider DST transition in analysis.
    THRESHOLD (float): Threshold value for activity filtering.

    XENO_CANTO_URL (str): Base URL for the Xeno-Canto API.

    RANDOM_SEED (int): Random seed for reproducibility.

    MILON (DataFrame): DataFrame containing species mapping data read from 'milon.txt'.
"""

import os
import pandas as pd

# Environment configuration
WORKING_ENV = 'work'  # Options: 'work', 'home', 'other'

# Setting paths based on the working environment
if WORKING_ENV == 'work':
    ROOT_PATH = r"H:\My Drive\לימודים\תואר שני\project bird recognition\python\AgmoNET"
    MILON_PATH = os.path.join(ROOT_PATH, 'resources')
    INPUT_DATA_PATH = r"H:\My Drive\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings"
    OUTPUT_DATA_PATH_LOC = r"C:\Users\basha\OneDrive\Desktop\offline data"
    SPECIES_ACTIVITY_PATH = r"H:\My Drive\לימודים\תואר שני\project bird recognition\species activity2"
    DF2_PATH = r""

elif WORKING_ENV == 'home':
    ROOT_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\AgmoNET"
    MILON_PATH = os.path.join(ROOT_PATH, 'resources')
    INPUT_DATA_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings"
    OUTPUT_DATA_PATH_LOC = r"C:\Users\basha\OneDrive\Desktop\offline data"
    SPECIES_ACTIVITY_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\species activity2"
    DF2_PATH = r"C:\Users\basha\OneDrive\Desktop\offline data\df2\station 2"

elif WORKING_ENV == 'other':
    ROOT_PATH = r""
    MILON_PATH = os.path.join(ROOT_PATH, 'resources')
    INPUT_DATA_PATH = r""
    OUTPUT_DATA_PATH_LOC = r""
    SPECIES_ACTIVITY_PATH = r""
    DF2_PATH = r""

else:
    ROOT_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\AgmoNET"
    MILON_PATH = os.path.join(ROOT_PATH, 'resources')
    INPUT_DATA_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings"
    OUTPUT_DATA_PATH_LOC = r"C:\Users\basha\OneDrive\Desktop\offline data"
    SPECIES_ACTIVITY_PATH = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\species activity2"
    DF2_PATH = r"C:\Users\basha\OneDrive\Desktop\offline data\df2\station 2"

# Data processing parameters
SR = 48000  # Sampling rate for audio processing
MIN_SAPLES = 50
HOP_LENGTH = 256  # Hop length for spectrogram computation
N_FFT = 1024  # Number of FFT components
MIN_FREQ = 200  # Minimum frequency for spectrogram analysis
MAX_FREQ = 15000  # Maximum frequency for spectrogram analysis
MEL = True  # Whether to use mel scale for spectrogram
N_MELS = 40  # Number of mel bands if MEL is True
SEGMENT_DURATION = 3  # Duration of each audio segment in seconds
OVERLAP = 0  # Overlap between audio segments in seconds
FACTOR = 3 # Factor for 
TRAIN_SIZE = 0.9  # Proportion of data used for training

# Model training parameters
N_AUG = 1  # Number of augmentations per signal for data augmentation
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 64  # Batch size for training
VALIDATION_SPLIT = 0.1  # Validation split ratio

# Activity analysis parameters
SPECIES_LIST = [457, 518, 320, 287, 286, 161, 377, 77, 138, 416, 42, 50, 158, 316, 411, 442, 191, 369, 418, 315, 431, 149]
STATIONS = ['2', '3']
DATE_START = '11-18-2020'  # Start date for analysis (format M-D-Y)
DATE_END = '11-22-2022'  # End date for analysis (format M-D-Y)
DST_TRANSITION = False  # Whether to consider DST transitions in analysis
THRESHOLD = 0.1  # Threshold value for activity filtering

# API configurations
XENO_CANTO_URL = "https://xeno-canto.org/"  # Base URL for the Xeno-Canto API

# General constants
RANDOM_SEED = 42  # Random seed for reproducibility

# Load the species mapping (milon) into a DataFrame
# MILON = pd.read_csv(os.path.join(MILON_PATH, "milon.txt"), sep="\t", encoding='iso8859_8')