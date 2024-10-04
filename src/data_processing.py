# -*- coding: utf-8 -*-
"""
data processing

@author: Moshe Bashan
"""

#%% IMPORT PACKAGES:
import glob, os
import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True
import librosa, librosa.display
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, GainTransition
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import time

#%% read_labels:
def read_labels(files_path,milon,save_files=False):
    """
    The read_labels function efficiently divides long audio recordings
    into labeled segments based on corresponding text files.
    It accepts WAV files and their associated text files with matching names.
    It is specifically designed for compatibility with Audacity software,
    accommodating both spectral and non-spectral selections.
       
    Input Parameters:
    - files_path: A string specifying the path to the directory containing the WAV files
    and their corresponding text files.
    - milon: A string specifying the path to the MILON file or a DataFrame/text/CSV file
    containing all classes (species) names, IDs, and labels.
    - save_files: An optional boolean argument (default: False).
    If set to True, it will save all the signals in the 'results' folder.
      
    Return Value:
    - samples: A DataFrame containing all the signals (optional to use).
    - labels_log: A Dataframe containing details for all labels.
      
    Example Usage:
    files_path = r"C:\projects\bird sound identification\labeled recordings" # data path
    milon = r"C:\projects\bird sound identification\supplementary" # milon path
    samples, labels_log = read_labels(milon_path, files_path, save_files=True)       
    """
    if isinstance(milon, str):
        os.chdir(milon)
        milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding=('iso8859_8')) # read dataframe with all species nams, id's and labels        
    os.chdir(files_path)
    if save_files == True and not os.path.isdir('results'):
        os.mkdir('results') # make new folder for results    
    wav_files = glob.glob("*.wav") # make list of all wav files
    #text_files = glob.glob("*.txt") # make list of all text files   
    samples = pd.DataFrame({'file name': [],'signal': []})
    labels_log = pd.DataFrame(columns=('file name', 'label', 'index', 'start time', 'end time', 'min freq', 'max freq')) # new dataframe for lables log
    labels_log_row = 0
    unidentified = pd.DataFrame(columns=('file name', 'index', 'lable')) # new dataframe for unidentified labels 
    unidentified_row = 0        
    for i in range(len(wav_files)): # loop over all files
        file_name = wav_files[i] # get the file name
        fname = file_name[0:len(file_name)-4] # get the file name without the suffix
        #signal, sr = librosa.load(file_name, sr=22050) # read the wav file
        text_file = pd.read_csv(fname + '.txt', sep="\t", header=None) # read the text file       
        if text_file.shape[0]<=1 or text_file[0][1] != '\\': # if spectral selection mode is disable or enable
            spectral_selection = 'disable'
            labels = text_file[2]
            time_start = text_file[0] # time start in sec
            time_end = text_file[1] # time end in sec
            labels_df = pd.concat([labels.reset_index(drop=True), time_start.reset_index(drop=True), 
                               time_end.reset_index(drop=True)], axis = 1)
            labels_df.columns = ['labels','time start','time end']
        else:
            spectral_selection = 'enable'
            labels = text_file[2][0::2].to_frame() # get all lables from the text file
            time_start = text_file[0][0::2].to_frame() # get the start time from all samples
            time_end = text_file[1][0::2].to_frame() # get the end time from all samples
            freq_min = text_file[1][1::2].to_frame() # get the minimum frequency from all samples
            freq_max = text_file[2][1::2].to_frame() # get the maximum frequency from all samples
            labels_df = pd.concat([labels.reset_index(drop=True), time_start.reset_index(drop=True), 
                               time_end.reset_index(drop=True), freq_min.reset_index(drop=True), freq_max.reset_index(drop=True)], axis = 1)
            labels_df.columns = ['labels','time start','time end','freq min','freq max']               
        for j in range(len(labels_df)): # loop over all lables in the file
            index = j+1 # index number of sample in the recording
            label = labels_df['labels'][j] # lable name (species)
            col = 0
            match = milon[milon[col] == label] # find match for lable from file with label table (in column=1)
            while col < np.shape(milon)[1]-1:  # look for label match in different column if label is unidentified
                col = col + 1
                if match.shape[0] == 0:
                    match = milon[milon[col] == label] # find match for lable from file with label table (in column=col)            
            if match.shape[0] == 0: # if label is unidentified
                unidentified = True
                unidentified.loc[unidentified_row] = [fname, index, label] # update unidentified table
                unidentified_row += 1
            elif match.shape[0] != 0: # if lable is identified:
                species_id = int(match[0]) # in file index lable
                new_file_name = str(species_id) + '_' + str(index) + '_' + file_name # new file name             
                samp_start = float(labels_df['time start'][j]) # start time of the sample in seconds
                samp_duration = float(labels_df['time end'][j]) - float(labels_df['time start'][j]) # duration of the sample in seconds
                sample, sr = librosa.load(wav_files[i], offset = samp_start, duration = samp_duration, sr = 44100) # read the audio signal                
                if sr != 44100: # resample to 44.100 kHz if otherwise 
                    librosa.resample(sample, sr, 44100)               
                # write the sample in the results DataFrame and folder :
                samples.loc[len(samples)] = (new_file_name,sample)
                if save_files == True:                    
                    os.chdir('results') # change to results folder
                    os.write(new_file_name,sample, samplerate = sr) # write the wav file
                    os.chdir(files_path) # change back to working directory                
                # update log:
                if spectral_selection == "enable": # update metadata with or without spectral selection
                    labels_log.loc[labels_log_row] = [fname,labels_df['labels'][j],index,labels_df['time start'][j],labels_df['time end'][j],labels_df['freq min'][j],labels_df['freq max'][j]]
                    labels_log_row += 1
                else:
                    labels_log.loc[labels_log_row] = [fname,labels_df['labels'][j],index,labels_df['time start'][j],labels_df['time end'][j],[],[]]
                    labels_log_row += 1       
    labels_log = [labels_log, unidentified] # make log for all samples include unidentified       
    if not unidentified.empty:
        print("unidentified labels:")
        print(unidentified)    
    print("======      End Of Process      ======")
    return samples, labels_log


#%% read_files:
def read_files(input_data_path, milon_path, sr=48000):
    """
    Reads and processes audio files from the specified input directory, extracting relevant metadata and audio signals.
    The function organizes this information into a pandas DataFrame, making it suitable for further analysis or modeling.

    Parameters:
        input_data_path (str): Path to the directory containing the input .wav audio files. This directory may contain subdirectories.
        milon_path (str): Path to the directory containing the 'milon.txt' file, which maps species index numbers to species names.
        sr (int): Sampling rate for loading the audio files.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'dataset': The name of the dataset (derived from the folder name).
            - 'file_name': The name of the .wav file.
            - 'class_id': The species index number extracted from the file name.
            - 'species_eng': The English name of the species, as per the 'milon.txt' file.
            - 'signal': The audio signal data loaded from the .wav file.
        dict: A dictionary named 'metadata' containing:
            - 'summary': A summary of the processed files, including total samples, total length (in minutes), total size (in MB), and dataset information.
            - 'species': A dictionary with species names as keys and their total sample counts across all datasets.
            - 'datasets': A dictionary detailing the count of species within each dataset.
            - 'details': A DataFrame with additional information about each file, including file length (in seconds) and size (in KB).

    Usage:
        data, metadata = read_files(input_data_path, milon_path, sr)   
    """
    print('STATUS: Reading files')
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding='iso8859_8')
    data = pd.DataFrame(columns=['dataset', 'file_name', 'class_id', 'species_eng', 'signal'])
    dataset_info = {}
    total_samples = 0
    total_length = 0.0
    total_size = 0
    file_counter = 0

    for root, dirs, files in os.walk(input_data_path):
        # Skip directories without .wav files
        wav_files = [file for file in files if file.endswith(".wav")]
        if not wav_files:
            continue

        dataset_name = os.path.basename(root)
        dataset_info[dataset_name] = {}

        for file in wav_files:
            file_path = os.path.join(root, file)
            print(f'Dataset: {dataset_name} File: {file}')
            signal, sr = librosa.load(file_path, sr=sr)
            length = len(signal) / sr
            size = os.path.getsize(file_path)
            total_samples += 1
            total_length += length
            total_size += size

            # Extract species index from the file name
            label_id = file.split('_')[0] if '_' in file else file[:3]  # Adapt this if the index extraction logic varies

            species_eng = milon[milon[0] == label_id][3].to_string(index=False).strip()
            data.loc[len(data)] = [dataset_name, file_path, label_id, species_eng, signal]

            # Update dataset and species count information
            dataset_info[dataset_name][species_eng] = dataset_info[dataset_name].get(species_eng, 0) + 1
            
        #     # Stop after processing 500 files 
        #     file_counter += 1  # Increment the counter
        #     if file_counter >= 1000:  # Stop after processing 100 files
        #         break
        # if file_counter >= 1000:  # Ensure loop exits after processing 100 files
        #     break

    # Prepare summary and species dictionaries
    summary = {
        'total_samples': total_samples,
        'total_length': total_length / 60,  # in minutes
        'total_size': total_size / (1024 * 1024),  # in MB
        'datasets': {dataset_name: sum(classes.values()) for dataset_name, classes in dataset_info.items()}
    }

    # Sum the total samples for each species across all datasets
    species = {}
    for classes in dataset_info.values():
        for species_name, count in classes.items():
            species[species_name] = species.get(species_name, 0) + count

    datasets = {name: pd.Series(classes).to_dict() for name, classes in dataset_info.items()}

    details = data[['dataset', 'file_name', 'class_id', 'species_eng']].copy()
    details['length'] = details['file_name'].apply(lambda x: librosa.get_duration(path=x))  # in seconds
    details['size'] = details['file_name'].apply(lambda x: os.path.getsize(x) / 1024)  # in KB

    metadata = {
        'summary': summary,
        'species': species,  # Updated to contain total sample counts across all datasets
        'datasets': datasets,
        'details': details,
    }

    print('STATUS: Reading files - Done')
    return data, metadata

#%% segment_data:   
def segment_data(data, segment_duration=3, sr=48000, overlap=0, method='wgn', factor=30):
    """
    Segments the data into intervals of a specified duration and produces metadata.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the dataset.
    segment_duration (int): Duration of each segment in seconds (default is 3 seconds).
    sr (int): Sample rate of the signal (default is 48000 samples/second).
    overlap (float): Congruence (overlap) between segments in fraction (0-1). Default is 0.
    method (str): Padding method ('wgn' for white Gaussian noise, '0' for zeros). Default is 'wgn'.
    factor (int): For 'wgn', this represents the SNR in dB; for '0', it represents the padding constant. Default is 30.

    Returns:
    segmented_data (pd.DataFrame): A DataFrame containing the segmented data.
    metadata (dict): Metadata dictionary containing various statistics.

    Note:
    - If the signal is shorter than the segment duration or has a leftover portion after segmenting, it will be padded with the specified method.

    Usage:
    segmented_data, segment_metadata = segment_data(data, segment_duration=3, sr=48000, overlap=0, method='wgn', factor=3)
    """
    print('STATUS: Applying segmentations')

    def seg_padding(x, seg_len, method='wgn', factor=factor):
        """
        Pads segments shorter than a minimum length required.
        
        Parameters:
        x (np.ndarray): Audio segment.
        seg_len (int): Required segment length in samples.
        method (str): Padding method ('wgn' for white Gaussian noise, '0' for zeros).
        factor (int): For 'wgn', this represents the SNR in dB; for '0', it represents the padding constant.
        
        Returns:
        np.ndarray: A padded segment.
        """
        x_len = x.shape[0]
        add_seg = seg_len - x_len
        seg_l = add_seg // 2
        seg_r = add_seg - seg_l  # To handle odd padding lengths
        
        if method == '0':
            y = np.pad(x, (seg_l, seg_r), 'constant', constant_values=(factor, factor))
        elif method == 'wgn':
            SNR_dB = factor
            sigma = np.sqrt(np.mean(x ** 2) / (10 ** (SNR_dB / 10)))
            y = np.pad(x, (seg_l, seg_r), mode='constant', constant_values=0)  # Pads around the original signal
            y[:seg_l] = sigma * np.random.randn(seg_l)  # Add WGN padding before
            y[-seg_r:] = sigma * np.random.randn(seg_r)  # Add WGN padding after                               
        return y
    
    segment_length = segment_duration * sr
    step = int(segment_length * (1 - overlap))
    segmented_data_list = []
    
    for idx, row in data.iterrows():
        signal = row['signal']
        num_samples = len(signal)
    
        for start in range(0, num_samples, step):
            end = start + segment_length
            if end > num_samples:
                segment = seg_padding(signal[start:], segment_length, method=method, factor=factor)
            else:
                segment = signal[start:end]
                
            new_row = [row['dataset'], row['file_name'], row['class_id'], row['species_eng'], segment]
            segmented_data_list.append(new_row)
            
    segmented_data = pd.DataFrame(segmented_data_list, columns=['dataset', 'file_name', 'class_id', 'species_eng', 'signal'])        
    
    # Generate metadata
    # Count samples before and after segmentation
    before_counts = data['species_eng'].value_counts().sort_index()
    after_counts = segmented_data['species_eng'].value_counts().sort_index()
    samples_df = pd.DataFrame({'before': before_counts, 'after': after_counts}).fillna(0)

    # Dataset-specific sample counts before and after segmentation
    datasets = data['dataset'].unique()
    datasets_samples = {}
    datasets_total_samples = {}
    for dataset in datasets:
        dataset_data = data[data['dataset'] == dataset]
        dataset_segmented_data = segmented_data[segmented_data['dataset'] == dataset]
        before_counts = dataset_data['species_eng'].value_counts().sort_index()
        after_counts = dataset_segmented_data['species_eng'].value_counts().sort_index()
        datasets_samples[dataset] = pd.DataFrame({'before': before_counts, 'after': after_counts}).fillna(0)
        datasets_total_samples[dataset] = {
            'before': dataset_data.shape[0],
            'after': dataset_segmented_data.shape[0]
        }
    
    # Calculate total samples before and after segmentation
    total_samples = {
        'before': samples_df['before'].sum(),
        'after': samples_df['after'].sum()
    }

    # Collect parameters
    parameters = {
        'segment_duration': segment_duration,
        'sr': sr,
        'overlap': overlap,
        'method': method,
        'factor': factor
    }

    # Combine into metadata dictionary
    metadata = {
        'samples': samples_df,
        'datasets_samples': datasets_samples,
        'datasets_total_samples': datasets_total_samples,
        'total_samples': total_samples,
        'parameters': parameters
    }
    print('STATUS: Applying segmentations - Done')
    return segmented_data, metadata
#%% split_dataset:
def split_dataset(data, train_size=0.9, stratify=True, min_samples=2):
    """
    Splits the dataset into training and testing sets, and produces metadata.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the dataset.
    train_size (float): The proportion of the dataset to include in the training set (default is 0.9).
    stratify (bool): If True, perform stratified sampling based on the 'class_id' column (default is True).
    min_samples (int): Minimum number of samples required per class. Classes with fewer samples will be removed (default is 2).

    Returns:
    train_data (pd.DataFrame): DataFrame containing the training set.
    test_data (pd.DataFrame): DataFrame containing the test set.
    y_to_species (dict): Mapping of class_id to species names.
    metadata (dict): Metadata dictionary containing various statistics.

    Usage:
    train_data, test_data, y_to_species, metadata = split_dataset(data, train_size=0.9, stratify=True, min_samples=4)
    """
    print('STATUS: Splitting datasets')
    if min_samples < 2:  # Ensure min_samples is at least 2
        print("min_samples was set below 2. Automatically setting it to 2.")
        min_samples = 2
    # Filter out classes with fewer than min_samples:
    class_counts = data['class_id'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    filtered_data = data[data['class_id'].isin(valid_classes)]    
    # Collect removed classes with species names
    removed_classes = class_counts[class_counts < min_samples]
    samples_removed = {
        data[data['class_id'] == class_id]['species_eng'].iloc[0]: count
        for class_id, count in removed_classes.items()}
    if samples_removed:
        print("Removed species with fewer than {} instances:".format(min_samples))
        for species, count in samples_removed.items():
            print(f"Species: {species}, Samples removed: {count}")
        
    filtered_data['y'] = pd.factorize(filtered_data['class_id'])[0] # Factorize 'class_id' to 'y' for modeling
    y_to_species = dict(enumerate(filtered_data[['class_id', 'species_eng']].drop_duplicates().set_index('class_id')['species_eng']))
    
    # Split the data into training and testing sets:
    if stratify:
        train_data, test_data = train_test_split(
            filtered_data, train_size=train_size, stratify=filtered_data['class_id'], random_state=42)
    else:
        train_data, test_data = train_test_split(
            filtered_data, train_size=train_size, random_state=42)
    
    # Metadata Calculation
    total_samples = {'train': len(train_data),'test': len(test_data)}    
    # Species sample counts for train and test
    species_train = train_data['species_eng'].value_counts().sort_index()
    species_test = test_data['species_eng'].value_counts().sort_index()
    species_samples = pd.DataFrame({'train': species_train, 'test': species_test}).fillna(0).astype(int)    
    # Dataset-specific sample counts for train and test
    datasets = data['dataset'].unique()
    datasets_samples = {}
    datasets_species = {}
    for dataset in datasets:
        dataset_train = train_data[train_data['dataset'] == dataset]
        dataset_test = test_data[test_data['dataset'] == dataset]
        datasets_samples[dataset] = {'train': len(dataset_train),'test': len(dataset_test)}
        train_species_count = dataset_train['species_eng'].value_counts().sort_index()
        test_species_count = dataset_test['species_eng'].value_counts().sort_index()
        datasets_species[dataset] = pd.DataFrame({'train': train_species_count, 'test': test_species_count}).fillna(0).astype(int)   
    # Collect input parameters
    parameters = {'train_size': train_size,'stratify': stratify,'min_samples': min_samples}    
    # Compile metadata
    metadata = {
        'total_samples': total_samples,
        'species': species_samples,
        'datasets_samples': datasets_samples,
        'datasets_species': datasets_species,
        'y_to_species': y_to_species,
        'samples_removed': samples_removed,
        'parameters': parameters}
        
    print('STATUS: Spliting datasets - Done')
    return train_data, test_data, y_to_species, metadata 
#%% augment_data: 
def augment_data(data, n_aug, sr=48000, balance_classes=True):
    """
    Perform random audio augmentations on signals using the audiomentations library.
    Applies AddGaussianNoise, TimeStretch, PitchShift, and combinations thereof.
    For each signal in the input DataFrame, the specified number of augmentations
    are applied unless balance_classes is True, in which case the number of augmentations
    will be adjusted to balance the number of samples across classes.

    Parameters:
    data (pandas.DataFrame): Input DataFrame with signals in the 'signal' column.
    n_aug (int): Minimum number of augmentations to apply per signal.
    sr (int): Sample rate (default is 48000).
    balance_classes (bool): Whether to balance the number of samples across classes (default is True).

    Returns:
    augmented_data (pandas.DataFrame): DataFrame containing both original and augmented signals.
    metadata (dict): Metadata dictionary containing augmentation statistics.

    Usage:
    augmented_data, metadata = augment_data(data, n_aug=2, sr=48000, balance_classes=True)
    """
    print('STATUS: Applying augmentations')

    augmentations = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1),
        TimeStretch(min_rate=0.75, max_rate=1.25, p=1),
        PitchShift(min_semitones=-3, max_semitones=3, p=1),
        Gain(min_gain_db=-5, max_gain_db=5, p=1),
        GainTransition(min_gain_db=-10, max_gain_db=10, p=1)
    ]

    # Initialize the augmented data with the original data
    augmented_data = data.copy()

    # Calculate species and dataset counts (unchanged throughout)
    species_counts = data['species_eng'].value_counts()
    dataset_counts = data['dataset'].value_counts()

    # Calculate the number of samples for each class
    class_counts = data['y'].value_counts()
    max_samples = class_counts.max()

    for class_ind, count in class_counts.items():
        # Filter data for the current class
        class_data = data[data['y'] == class_ind]
        species_name = class_data['species_eng'].iloc[0]  # Assuming species_eng column exists

        for _, row in class_data.iterrows():
            signal = row['signal']
            
            # Calculate required samples and number of augmentations
            current_class_samples = len(class_data)
            required_samples = max(int((max_samples * (n_aug + 1)) * 0.5), (n_aug + 1) * current_class_samples)
            num_augmentations = int(np.floor((required_samples / current_class_samples) - 1))

            # Apply the calculated number of augmentations or the minimum specified
            for _ in range(max(num_augmentations, n_aug)):
                n_combinations = random.randint(1, len(augmentations))  # Random number of augmentations 
                selected_augmentations = random.sample(augmentations, n_combinations)  # Random augmentations
                augment = Compose(selected_augmentations)
                augmented_signal = augment(samples=signal, sample_rate=sr)
                
                # Create a new row with the augmented signal
                new_row = row.copy()
                new_row['signal'] = augmented_signal
                
                # Concatenate the new row to the augmented_data DataFrame
                augmented_data = pd.concat([augmented_data, pd.DataFrame([new_row])], ignore_index=True)

    # Calculate the final counts after augmentations
    final_species_counts = augmented_data['species_eng'].value_counts()
    final_dataset_counts = augmented_data['dataset'].value_counts()

    # Calculate the number of augmentations made per species
    augmentations_per_species = {
        species: max(0, (final_species_counts[species] - species_counts[species]) // species_counts[species])
        for species in species_counts.index
    }

    # Creating metadata dictionary
    metadata = {
        'augmentations': augmentations_per_species,
        'total_samples': {
            'before': len(data),
            'after': len(augmented_data)
        },
        'species': {
            species: {
                'before': species_counts[species],
                'after': final_species_counts.get(species, 0)
            }
            for species in species_counts.index
        },
        'datasets': {
            dataset: {
                'before': dataset_counts[dataset],
                'after': final_dataset_counts.get(dataset, 0)
            }
            for dataset in dataset_counts.index
        },
        'datasets_species': {},
        'parameters': {
            'n_aug': n_aug,
            'sr': sr,
            'balance_classes': balance_classes
        }
    }

    # Calculate dataset-species specific samples before and after augmentations
    datasets = data['dataset'].unique()
    for dataset in datasets:
        dataset_before = data[data['dataset'] == dataset]['species_eng'].value_counts()
        dataset_after = augmented_data[augmented_data['dataset'] == dataset]['species_eng'].value_counts()
        metadata['datasets_species'][dataset] = {
            species: {
                'before': dataset_before.get(species, 0),
                'after': dataset_after.get(species, 0)
            }
            for species in dataset_before.index.union(dataset_after.index)
        }

    print('STATUS: Applying augmentations - Done')
    return augmented_data, metadata


#%% to_spec: 
def to_spec(data,sr=48000,hop_length=256,n_fft=1024,mel=True,n_mels=40,min_freq=200,max_freq=15000):
    """
    Convert all signals in the DataFrame to spectrograms or mel-spectrograms,
    replacing the 'signal' column with the spectrogram data. Also generates metadata
    about the dataset.

    Parameters:
    data (pandas.DataFrame): Input DataFrame with signals in the 'signal' column.
    sr (int): Sample rate of the audio signals. Default is 48000.
    hop_length (int): Number of samples between frames. Default is 256.
    n_fft (int): Number of samples per frame for FFT. Default is 1024.
    min_freq (int): Minimum frequency to display on the y-axis. Default is 200.
    max_freq (int): Maximum frequency to display on the y-axis. Default is 15000.
    mel (bool): If True, compute mel-spectrogram instead of a standard spectrogram. Default is True.
    n_mels (int): Number of mel bands to generate. Used only if mel=True. Default is 128.

    Returns:
    tuple: 
        - pandas.DataFrame: DataFrame with 'signal' column replaced by spectrogram data.
        - dict: Metadata dictionary with information about the dataset and processing parameters.
   
    Usage:
    spec_data, final_metadata = to_spec(augmented_data,sr=48000,hop_length=256,n_fft=1024,mel=True,n_mels=40,min_freq=200,max_freq=15000)
    """
    print('STATUS: Converting to spectrogram')
    spec_data = data.copy()
    for idx, row in spec_data.iterrows():
        signal = row['signal']
        if mel:
            S = librosa.feature.melspectrogram(y=signal,sr=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,fmin=min_freq,fmax=max_freq)
            S_db = librosa.power_to_db(S)
        else:  # Compute standard spectrogram:
            S = np.abs(librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft))
            S_db = librosa.amplitude_to_db(S)            
        spec_data.at[idx, 'signal'] = S_db # Replace the signal with the computed spectrogram
    # Generate metadata
    total_samples = len(data)
    datasets_count = data['dataset'].value_counts().to_dict()

    # Create a nested dictionary for datasets_species
    datasets_species = {}
    for dataset, group in data.groupby('dataset'):
        species_count = group['species_eng'].value_counts().to_dict()
        datasets_species[dataset] = species_count

    species = data['species_eng'].value_counts().to_dict()
    parameters = {
        'sr': sr,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'mel': mel,
        'n_mels': n_mels,
        'min_freq': min_freq,
        'max_freq': max_freq
    }

    metadata = {
        'total_samples': total_samples,
        'datasets_count': datasets_count,
        'datasets_species': datasets_species,
        'species': species,
        'parameters': parameters
    }
    print('STATUS: Converting to spectrogram - Done')
    return spec_data, metadata

#%% process_for_analysis:
# function to pre-process one or multiple files before prediction
# maybe also for long term analysis (df1 and df2)

#%% birdnet_prep
def birdnet_prep(input_data_path,milon_path,output_data_path_loc,output_dir_name='results',window_size_sec=3,sr=48000):
    """
    birdnet_prep:
    Prepares audio data for use with BirdNET by organizing and processing
    audio files into species-specific directories. The function handles cases
    where audio files are shorter than a specified window size by padding the
    audio with either zeros or white Gaussian noise (WGN).

    Parameters:
        input_data_path (str): Path to the directory containing the input .wav audio files. This directory can optionally contain subdirectories.
        milon_path (str): Path to the directory containing the 'milon.txt' file, which maps species index numbers to species names.
        output_data_path_loc (str): Path to the directory where the processed audio files will be saved.
        output_dir_name (str, optional): Name of the directory where the results will be saved. Defaults to 'results'.
        window_size_sec (int, optional): Minimum duration (in seconds) for each audio file. Files shorter than this will be padded. Defaults to 3 seconds.
        sr (int, optional): Sampling rate for loading and saving the audio files. Defaults to 48000 Hz.

    Returns:
        None

    Usage:
        birdnet_prep(input_data_path,milon_path,output_data_path_loc)       
    """
    
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding=('iso8859_8')) # read species index numbers (label table)
    
    def wgn_sigma(x, SNR_dB):
        """
        Computing the amplitude of a random gaussian noise with a predefined SNR
        Input - x - the input signal to which the noise should be added
                SNR_dB - a required SNR
        returns y - the noise amplitude (or sigma)
        """
        L = x.size
        Es = (1 / L) * np.sum(x ** 2)
        SNR_lin = 10 ** (SNR_dB / 10)
        y = np.sqrt(Es / SNR_lin)
        return y
    
    def seg_padding(x, seg_len, method = 'wgn', factor = 1):
        """
        Padding segments shorter than a minimum length required
        Input arguments:
            x - audio segment, fs - sampling frequency, seg_len - required duration
            factor - if method = '0' - the padding constant, if method = 'wgn'
        return y - a padded segment
    
        """
        x_len = x.shape[0]
        add_seg = seg_len - x_len
        seg_l = add_seg // 2
        #seg_r = x_len - seg_l
        if method == '0':
            y = np.pad(x, (seg_l, seg_l), 'constant', constant_values=(factor, factor))
        if method == 'wgn':
            SNR_dB = factor
            sigma = wgn_sigma(x, SNR_dB)
            y = sigma * np.random.randn(seg_len)
            y[seg_l:seg_l + x_len] = x                        
        return y
    
    output_data_path = output_data_path_loc+"\\"+output_dir_name
    if not os. path. isdir(output_data_path):
        os.mkdir(output_data_path)
    subdir = [x[0] for x in os.walk(input_data_path)]
    if len(subdir) <= 1: # if dir is not contain subdirs
        os.chdir(input_data_path)
        wav_files = glob.glob("*.wav")
        for i in range(len(wav_files)):
            print("file:",i+1,"/",len(wav_files)+1)
            os.chdir(input_data_path)
            signal, sr = librosa.load(wav_files[i], sr = sr)
            if signal.shape[0]/sr < 3: # if signal is shorter then window size
                signal = seg_padding(signal, window_size_sec*sr, method = 'wgn', factor = 30)
            label_id = wav_files[i][0:3] # get the first three digits from file name
            if label_id [1] == '_': # for cases that the index number is in one or two digits
                label_id = label_id[0:1]
            elif label_id[2] == '_':
                label_id = label_id[0:2]          
            species_eng = milon[milon[0] == label_id][3].to_string(index=False) # get species name
            if os. path. isdir(output_data_path+"\\"+species_eng):
                os.chdir(output_data_path+"\\"+species_eng)
                sf.write(wav_files[i],signal, samplerate = sr) # write the wav file - (output)
            else:
                os.mkdir(output_data_path+"\\"+species_eng)
                os.chdir(output_data_path+"\\"+species_eng)
                sf.write(wav_files[i],signal, samplerate = sr) # write the wav file - (output)
    elif len(subdir) > 1: # if dir contain subdirs
        for sd in range(1,len(subdir)):
            os.chdir(subdir[sd])
            wav_files = glob.glob("*.wav") 
            for i in range(len(wav_files)):
                print("dir: ",sd,"/",len(subdir)-1,"file:",i+1,"/",len(wav_files)+1)
                os.chdir(subdir[sd])
                signal, sr = librosa.load(wav_files[i], sr = sr)
                if signal.shape[0]/sr < 3: # if signal is shorter then window size
                    signal = seg_padding(signal, window_size_sec*sr, method = 'wgn', factor = 30)            
                label_id = wav_files[i][0:3] # get the first three digits from file name
                if label_id [1] == '_': # for cases that the index number is in one or two digits
                    label_id = label_id[0:1]
                elif label_id[2] == '_':
                    label_id = label_id[0:2]
                species_eng = milon[milon[0] == species_eng][3].to_string(index=False) # get species name
                if os. path. isdir(output_data_path+"\\"+species_eng):
                    os.chdir(output_data_path+"\\"+species_eng)
                    sf.write(wav_files[i],signal, samplerate = sr) # write the wav file - (output)
                else:
                    os.mkdir(output_data_path+"\\"+species_eng)
                    os.chdir(output_data_path+"\\"+species_eng)
                    sf.write(wav_files[i],signal, samplerate = sr) # write the wav file - (output)
                    
    #sf.write(wav_files[i], wav_files[i],  sr) 


