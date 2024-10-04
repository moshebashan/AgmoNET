# -*- coding: utf-8 -*-
"""
Main script
@Author: Moshe Bashan
"""

#%% import packages and set parameters

# Import packages:  
import os
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# Environment configuration
# WORKING_ENV = 'home'  # Options: 'work', 'home', 'other'

# SET PATH's
working_from = 'home' # options: 'work','home','other'
if working_from == 'work':
    root_path = r"H:\My Drive\לימודים\תואר שני\project bird recognition\python\AgmoNET"
    milon_path = os.path.join(root_path, 'resources')
    input_data_path = r"H:\My Drive\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings"
    output_data_path_loc = r"C:\Users\basha\OneDrive\Desktop\offline data"
    species_activity_path = r"H:\My Drive\לימודים\תואר שני\project bird recognition\python\AgmoNET\data\output\species_activity_data" 
    df2_path = r""
    species_activity_data_path = os.path.join(root_path, 'data\output\species_activity_data')            
elif working_from == 'home':
    root_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\AgmoNET"
    milon_path = os.path.join(root_path, 'resources')
    input_data_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings"
    output_data_path_loc = r"C:\Users\basha\OneDrive\Desktop\offline data" 
    species_activity_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\AgmoNET\data\output\species_activity_data" 
    df2_path = r"C:\Users\basha\OneDrive\Desktop\offline data\df2"
    species_activity_data_path = os.path.join(root_path, 'data\output\species_activity_data')      
elif working_from == 'other': # set path for Yizhar or other users
    root_path =  r""
    milon_path = r""
    input_data_path = r""
    output_data_path_loc = r""
    pecies_activity_path = r""
    df2_path = r""

os.chdir(root_path)
# Local application imports
from src.data_processing import read_files, segment_data, split_dataset, augment_data, to_spec
from src.modeling import build_compile, plot_history, simple_resnet, model_evaluate
from src.activity_analysis import (
    activity_levels_comparison, activity_distribution_overview,
    species_activity_aggregator, handle_missing_data, species_level_overview,
    habitat_indices, habitat_indices, activity_image)
from src.visualization import (plot_soundwaves_and_spectrogram, visual_datasets,
    visual_spec, display_segment)
from src.utils import play_sound

from src.config import (
    SR, HOP_LENGTH, N_FFT, MIN_FREQ, MAX_FREQ, MEL, 
    N_MELS, MIN_SAPLES, SEGMENT_DURATION, OVERLAP, 
    N_AUG, SPECIES_LIST, THRESHOLD, FACTOR, TRAIN_SIZE,
    STATIONS, DATE_START, DATE_END)

#species_list = [418,431,416,286,149,377,314,191,320,457,369,492,138,171,304,315] # old list
# species_list = [457,518,320,287,286,161,377,77,138,416,42,50,158,316,411,442,191,369,418,315,431,149]
species_list = [457,161,377,77,158,191,315,320,286,138,316,411,369,518,287,42,50,369,431,149,416,442]
dst_transition=False
   
# Set parameters
# Data processing parameters
# MIN_SAPLES = 50
# HOP_LENGTH = 256  # Hop length for spectrogram computation
# N_FFT = 1024  # Number of FFT components
# N_MELS = 40  # Number of mel bands if MEL is True
# OVERLAP = 0  # Overlap between audio segments in seconds

# Model training parameters
# N_AUG = 5  # Number of augmentations per signal for data augmentation
# EPOCHS = 50  # Number of training epochs
# BATCH_SIZE = 64  # Batch size for training

# Activity analysis parameters
# SPECIES_LIST = [457, 518, 320, 287, 286, 161, 377, 77, 138, 416, 42, 50, 158, 316, 411, 442, 191, 369, 418, 315, 431, 149]
# THRESHOLD = 0.1  # Threshold value for activity filtering

# Load the species mapping (milon) into a DataFrame
# MILON = pd.read_csv(os.path.join(MILON_PATH, "milon.txt"), sep="\t", encoding='iso8859_8')

# Import milon:
# pickle_path = os.path.join(root_path, 'resources/milon.pkl')
# with open(pickle_path, 'rb') as f:
#     milon = pickle.load(f)


#%% Pre-proccess:
# Read the data   
data, files_metadata = read_files(input_data_path,milon_path,sr = SR,)
play_sound
# Segment the data
segmented_data, segment_metadata = segment_data(data,segment_duration=SEGMENT_DURATION,sr=SR,overlap=OVERLAP,method='wgn',factor=FACTOR,) 
# Split the data for train and test
train_data, test_data, y_to_species, split_metadata = split_dataset(
    segmented_data,
    train_size=TRAIN_SIZE,
    stratify=True,
    min_samples=50
    )
# 
augmented_train_data, augment_metadata = augment_data(
    train_data,
    n_aug=1,
    sr=SR,
    balance_classes=True
    )
play_sound()

# augmented_train_data, augment_metadata = augment_data(
#     train_data,
#     n_aug=N_AUG,
#     sr=SR,
#     balance_classes=True
#     )
# convert signal to spectrogram
spec_train_data, train_metadata = to_spec(augmented_train_data,sr=SR,hop_length=HOP_LENGTH,n_fft=N_FFT,min_freq=MIN_FREQ,max_freq=MAX_FREQ,mel=MEL,n_mels=N_MELS)  
spec_test_data, test_metadata = to_spec(test_data,sr=SR,hop_length=HOP_LENGTH,n_fft=N_FFT,min_freq=MIN_FREQ,max_freq=MAX_FREQ,mel=MEL,n_mels=N_MELS)

# save and load dataset:
# Creating the metadata dictionary
metadata = {
    'files_metadata': files_metadata,
    'segment_metadata': segment_metadata,
    'split_metadata': split_metadata,
    'augment_metadata': augment_metadata,
    'train_metadata': train_metadata,
    'test_metadata': test_metadata
    }
# Creating the dataset dictionary
dataset = {'train_data': spec_train_data,
           'test_data': spec_test_data,
           'metadata': metadata
           }

# Save dataset
pickle.dump(dataset, open(os.path.join(root_path, 'data\datasets\min_dataset.pkl'), 'wb')) # save dataset

# Load dataset
# load_dataset_name = 'dataset.pkl'
data_dict = pickle.load(open(os.path.join(root_path, 'data\datasets\min_dataset.pkl'), 'rb')) # load dataset
data_dict = pickle.load(open(r"C:\Users\basha\OneDrive\Desktop\offline data\full_dataset.pkl", 'rb')) # load dataset
data_dict = pickle.load(open(r"C:\Users\basha\OneDrive\Desktop\offline data\min_dataset.pkl", 'rb'))



##%% Model build & run:
model, X, y = build_compile(spec_train_data, model_function=simple_resnet) # model build & compile
model, X, y = build_compile(data_dict['train_data']) # model build & compile
model_history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1)
plot_history(model_history) # plot history

#%% Save and load trained model:
# Save model:
model.save(os.path.join(output_data_path_loc,'min_model.h5'))

# Load model:
#model = keras.models.load_model('path/to/your_model.h5')
model = keras.models.load_model(os.path.join(root_path, 'data\models\min_model.h5'))
"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\agmonet\data\output\models\min_model.h5"

#%% model evaluate:
X_test = np.moveaxis(np.stack(data_dict['test_data']['signal'], axis=1), 0, 1) # stacking list of 2D arrays to 3D array and changing axis
# X_test = np.moveaxis(np.stack(spec_test_data['signal'], axis=1), 0, 1) # stacking list of 2D arrays to 3D array and changing axis
X_test = X_test[...,np.newaxis] # add new axis for CNN models
y_test = data_dict['test_data']['y'].values
# y_test = spec_test_data['y'].values
evaluation_results = model_evaluate(model, X_test, y_test, y_to_species)
# evaluation_results = model_evaluate(model, X_test, y_test, y_to_species)

#%% long-term analysis:
# ***build function to make df1 and df2


# Make species activity dataframes and save to .xlsx files
species_activity_data_2 = species_activity_aggregator(milon_path,os.path.join(df2_path,'station 2'),species_list,threshold=THRESHOLD)
species_activity_data_3 = species_activity_aggregator(milon_path,os.path.join(df2_path,'station 3'),species_list,threshold=THRESHOLD)

# Remove missing dates from activity dataframes and creat missing_data.xlsx
missing_data = handle_missing_data(species_activity_data_path)

# Calculate total species activity difference between station and plot results
activity_levels_comparison(milon_path,species_activity_path,species_list=SPECIES_LIST,stations=STATIONS)

# Activity_distribution_overview
activity_distribution = activity_distribution_overview(milon_path, species_activity_path, species_list=SPECIES_LIST, date_start=DATE_START, date_end=DATE_END, stations=STATIONS)

# habitat_indices
habitat_indices(milon_path, species_activity_path, species_list=SPECIES_LIST, date_start=DATE_START, date_end=DATE_END, stations=STATIONS)

# Species_level_overview
total_summary = species_level_overview(milon_path,species_activity_path,species_list=SPECIES_LIST,date_start=DATE_START,date_end=DATE_END,stations=STATIONS)

# activity_images
peilut_all = activity_image(milon_path,species_activity_path,species_list=SPECIES_LIST,stations=STATIONS)

# daily_activity_changes

#%% Visualizations:
# plot soundwaves and spectrogram
plot_soundwaves_and_spectrogram([500, 2200, 4300], min_freq=0, max_freq=5000)

# visual datasets 
files_metadata, summary_table, species_count_table = visual_datasets(input_data_path, milon_path, sr=48000)

# display agumentations

# display segmentation
display_segment(input_data_path, segment_duration=3, sr=48000, overlap=0, method='wgn', factor=30, hop_length=256, n_fft=1024)


# display ...








