# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 23:58:17 2024

@author: basha
"""
import glob, os
import glob, os
import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True
import librosa, librosa.display
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import soundfile as sf
from datetime import date, datetime, timedelta
from scipy.signal import spectrogram
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from matplotlib.gridspec import GridSpec
import wave
import contextlib
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pydub import AudioSegment
#import time

#%% plot_soundwaves_and_spectrogram:
def plot_soundwaves_and_spectrogram(frequencies, duration=0.05, sampling_rate=44100, min_freq=None, max_freq=None):
    """
    Plots three individual soundwaves, their combination, and the spectrogram of the combined soundwave.
    
    Parameters:
    - frequencies: List of 3 frequencies for the individual soundwaves.
    - duration: Duration of the sound in seconds (default is 1 second).
    - sampling_rate: Number of samples per second (default is 44100 Hz).
    - min_freq: Minimum frequency for the spectrogram plot (optional).
    - max_freq: Maximum frequency for the spectrogram plot (optional).
    
    Usage:
    plot_soundwaves_and_spectrogram([500, 2200, 4300], min_freq=0, max_freq=5000)
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the individual sound waves (sine waves)
    wave1 = np.sin(2 * np.pi * frequencies[0] * t)
    wave2 = np.sin(2 * np.pi * frequencies[1] * t)
    wave3 = np.sin(2 * np.pi * frequencies[2] * t)

    # Combine the sound waves
    combined_wave = wave1 + wave2 + wave3
    
    # Compute the spectrogram of the combined sound wave
    f, t_spec, Sxx = spectrogram(combined_wave, fs=sampling_rate)

    # Filter the frequencies for plotting between min_freq and max_freq
    if min_freq is not None or max_freq is not None:
        freq_mask = np.ones(f.shape, dtype=bool)  # Create a mask for frequency filtering
        if min_freq is not None:
            freq_mask &= f >= min_freq  # Keep frequencies above min_freq
        if max_freq is not None:
            freq_mask &= f <= max_freq  # Keep frequencies below max_freq
        f = f[freq_mask]  # Apply the mask
        Sxx = Sxx[freq_mask, :]  # Filter the spectrogram data

    # Plot the three individual soundwaves
    plt.figure(figsize=(12, 8))

    plt.subplot(5, 1, 1)
    plt.plot(t, wave1, label=f"{frequencies[0]} Hz")
    plt.title(f"Soundwave 1: {frequencies[0]} Hz")
    plt.ylabel("Amplitude")

    plt.subplot(5, 1, 2)
    plt.plot(t, wave2, label=f"{frequencies[1]} Hz")
    plt.title(f"Soundwave 2: {frequencies[1]} Hz")
    plt.ylabel("Amplitude")

    plt.subplot(5, 1, 3)
    plt.plot(t, wave3, label=f"{frequencies[2]} Hz")
    plt.title(f"Soundwave 3: {frequencies[2]} Hz")
    plt.ylabel("Amplitude")

    # Plot the combined soundwave
    plt.subplot(5, 1, 4)
    plt.plot(t, combined_wave, label="Combined Wave", color='purple')
    plt.title("Combined Soundwave")
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")
    
    # Plot the spectrogram
    # plt.subplot(5, 1, 5)
    # plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    # plt.title("Spectrogram of Combined Soundwave")
    # plt.ylabel("Frequency [Hz]")
    # plt.xlabel("Time [s]")
    # plt.colorbar(label='Intensity [dB]')

    plt.tight_layout()
    plt.show()

    # Plot the spectrogram
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title("Spectrogram of Combined Soundwave")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.colorbar(label='Intensity [dB]')
    plt.show()


#%% visual datasets:
def visual_datasets(input_data_path, milon_path, sr=48000):
    """
    Function to process and visualize datasets with species information, sample length, and summary statistics.
    
    Usage:
    files_metadata, summary_table, species_count_table = visual_datasets(input_data_path, milon_path, sr=48000) 
    """
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding='iso8859_8')
    milon_dict = dict(zip(milon[0], milon[3]))  # key: label_id, value: species_eng
    files_metadata_list = []

    print('INFO: Reading datasets files information')
    for root, dirs, files in os.walk(input_data_path):
        wav_files = [file for file in files if file.endswith(".wav")]
        if not wav_files:
            continue
        dataset_name = os.path.basename(root)
    
        for f, file in enumerate(wav_files, 1):
            total_files = len(wav_files)
            print(f'\rINFO: Processing - Dataset: {dataset_name} File: {f}/{total_files}', end='', flush=True)
            file_path = os.path.join(root, file)
            with sf.SoundFile(file_path) as audio_file:
                length = round(len(audio_file) / audio_file.samplerate, 3)
            
            label_id = file.split('_')[0] if '_' in file else file[:3]
            species_eng = milon[milon[0] == label_id][2].to_string(index=False).replace(" ", "_")
            
            # Extract date and hour for the 'Agmon' dataset
            if dataset_name == 'Agmon':
                file_parts = file.split('_')
                date = datetime.strptime(file_parts[3][0:4]+'-'+file_parts[3][4:6]+'-'+file_parts[3][6:8], '%Y-%m-%d').date()
                hour = file_parts[4][:2]
                files_metadata_list.append([dataset_name, file, species_eng, length, date, hour])

    # Create DataFrame with new columns for date and hour
    files_metadata = pd.DataFrame(files_metadata_list, columns=['dataset', 'file_name', 'species_eng', 'length', 'date', 'hour'])

    # Figure 1: General distribution plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Subplot 1: Histogram of all samples' lengths
    axs[0].hist(files_metadata['length'], bins=50, color='blue', edgecolor='black')
    axs[0].set_title('Distribution of All Samples Length')
    axs[0].set_xlabel('Length (seconds)')
    axs[0].set_ylabel('Number of samples')
    axs[0].set_xlim(0, 10.1)

    # Subplot 2: Table of species sample count per dataset
    species_count_table = files_metadata.pivot_table(index='species_eng', columns='dataset', aggfunc='size', fill_value=0)
    species_count_table['Total'] = species_count_table.sum(axis=1)

    axs[1].axis('tight')
    axs[1].axis('off')
    table = axs[1].table(cellText=species_count_table.values, colLabels=species_count_table.columns, rowLabels=species_count_table.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()

    # Figure 2: Agmon-specific histograms
    agmon_data = files_metadata[files_metadata['dataset'] == 'Agmon']

    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 18))

    # Subplot 1: Histogram of samples' lengths for Agmon dataset
    axs2[0].hist(agmon_data['length'], bins=50, color='green', edgecolor='black')
    axs2[0].set_title('Distribution of Samples Length for Agmon Dataset')
    axs2[0].set_xlabel('Length (seconds)')
    axs2[0].set_ylabel('Number of samples')

    # Subplot 2: Histogram of number of samples from each date
    agmon_data['date'].dropna(inplace=True)
    axs2[1].hist(agmon_data['date'], bins=len(agmon_data['date'].unique()), color='orange', edgecolor='black')
    axs2[1].set_title('Number of Samples per Date (Agmon Dataset)')
    axs2[1].set_xlabel('Date')
    axs2[1].set_ylabel('Number of samples')
    axs2[1].tick_params(axis='x', rotation=45)

    # Subplot 3: Histogram of number of samples from each hour
    agmon_data['hour'].dropna(inplace=True)
    axs2[2].hist(agmon_data['hour'], bins=24, color='purple', edgecolor='black', range=(0, 24))
    axs2[2].set_title('Number of Samples per Hour (Agmon Dataset)')
    axs2[2].set_xlabel('Hour of the day')
    axs2[2].set_ylabel('Number of samples')

    plt.tight_layout()
    plt.show()

    # Figure 3: Summary table
    dataset_summary = files_metadata.groupby('dataset').agg(
        species_count=('species_eng', 'nunique'),
        sample_count=('file_name', 'size'),
        length_mean=('length', 'mean'),
        length_std=('length', 'std'),
        total_length=('length', 'sum')
    )
    
    # Add 'Total' row
    all_summary = pd.DataFrame({
        'species_count': [files_metadata['species_eng'].nunique()],
        'sample_count': [files_metadata['file_name'].size],
        'length_mean': [round(files_metadata['length'].mean(), 3)],
        'length_std': [round(files_metadata['length'].std(), 3)],
        'total_length': [files_metadata['length'].sum()]
    }, index=['Total'])
    
    summary_table = pd.concat([dataset_summary, all_summary])

    # Convert total length from seconds to minutes
    summary_table['total_length'] = round(summary_table['total_length'] / 60, 3)

    # Ensure integer type for species_count and sample_count in dataset_summary and summary_table
    summary_table = summary_table.astype({'species_count': 'int', 'sample_count': 'int'})

    # Plot table
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=summary_table.values, 
                      colLabels=summary_table.columns, 
                      rowLabels=summary_table.index, 
                      loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()
    
    print('INFO: visual datasets - Done')
    
    return files_metadata, summary_table, species_count_table
    
    
    
def visual_datasets1(input_data_path, milon_path, sr=48000):
    """
    


    Usage:
    files_metadata, summary_table = visual_datasets(input_data_path, milon_path, sr=48000) 
    """
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding='iso8859_8')
    milon_dict = dict(zip(milon[0], milon[3]))  # key: label_id, value: species_eng
    files_metadata_list = []

    print('INFO: Reading datasets files information')
    for root, dirs, files in os.walk(input_data_path):
        wav_files = [file for file in files if file.endswith(".wav")]
        if not wav_files:
            continue
        dataset_name = os.path.basename(root)
    
        for f, file in enumerate(wav_files, 1):
            total_files = len(wav_files)
            print(f'\rINFO: Processing - Dataset: {dataset_name} File: {f}/{total_files}', end='', flush=True)
            file_path = os.path.join(root, file)
            with sf.SoundFile(file_path) as audio_file:
                length = round(len(audio_file) / audio_file.samplerate,3)
            
            label_id = file.split('_')[0] if '_' in file else file[:3]
            species_eng = milon[milon[0] == label_id][2].to_string(index=False).replace(" ", "_")
            # species_eng = milon_dict.get(label_id, 'Unknown')
            
            # Extract date and hour for the 'Agmon' dataset
            if dataset_name == 'Agmon':
                # if file[3] == '_':
                #     ind_add = 1
                # elif file[2] == '_': 
                #     ind_add = 0
                
               file_parts = file.split('_')
               date = file_parts[3][6:8]
               date = datetime.strptime(file_parts[3][0:4]+'-'+file_parts[3][4:6]+'-'+file_parts[3][6:8], '%Y-%m-%d').date()
               hour = file_parts[4][:2]
               files_metadata_list.append([dataset_name, file, species_eng, length, date, hour])
                
                
                ##########
                    
                # try:
                #     date = datetime.strptime(file[15:19]+'-'+file[19:21]+'-'+file[21:23], '%Y-%m-%d').date()
                #     hour = int(file[25:27])
                # except (ValueError, IndexError):
                #     date, hour = None, None
            # else:
            #     date, hour = None, None
            
            
    
    # Create DataFrame with new columns for date and hour
    files_metadata = pd.DataFrame(files_metadata_list, columns=['dataset', 'file_name', 'species_eng', 'length', 'date', 'hour'])

    # Figure 1: General distribution plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Subplot 1: Histogram of all samples' lengths
    axs[0].hist(files_metadata['length'], bins=50, color='blue', edgecolor='black')
    axs[0].set_title('Distribution of All Samples Length')
    axs[0].set_xlabel('Length (seconds)')
    axs[0].set_ylabel('Number of samples')
    axs[0].set_xlim(0,10.1)

    # Subplot 2: Table of species sample count per dataset
    species_count = files_metadata.pivot_table(index='species_eng', columns='dataset', aggfunc='size', fill_value=0)
    species_count['Total'] = species_count.sum(axis=1)
    
    axs[1].axis('tight')
    axs[1].axis('off')
    table = axs[1].table(cellText=species_count.values, colLabels=species_count.columns, rowLabels=species_count.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()

    # Figure 2: Agmon-specific histograms
    agmon_data = files_metadata[files_metadata['dataset'] == 'Agmon']

    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 18))

    # Subplot 1: Histogram of samples' lengths for Agmon dataset
    axs2[0].hist(agmon_data['length'], bins=50, color='green', edgecolor='black')
    axs2[0].set_title('Distribution of Samples Length for Agmon Dataset')
    axs2[0].set_xlabel('Length (seconds)')
    axs2[0].set_ylabel('Number of samples')

    # Subplot 2: Histogram of number of samples from each date
    agmon_data['date'].dropna(inplace=True)
    axs2[1].hist(agmon_data['date'], bins=len(agmon_data['date'].unique()), color='orange', edgecolor='black')
    axs2[1].set_title('Number of Samples per Date (Agmon Dataset)')
    axs2[1].set_xlabel('Date')
    axs2[1].set_ylabel('Number of samples')
    axs2[1].tick_params(axis='x', rotation=45)

    # Subplot 3: Histogram of number of samples from each hour
    agmon_data['hour'].dropna(inplace=True)
    axs2[2].hist(agmon_data['hour'], bins=24, color='purple', edgecolor='black', range=(0, 24))
    axs2[2].set_title('Number of Samples per Hour (Agmon Dataset)')
    axs2[2].set_xlabel('Hour of the day')
    axs2[2].set_ylabel('Number of samples')

    plt.tight_layout()
    plt.show()

    # Figure 3: Summary table
    dataset_summary = files_metadata.groupby('dataset').agg(
        species_count=('species_eng', 'nunique'),
        sample_count=('file_name', 'size'),
        length_mean=('length', 'mean'),
        length_std=('length', 'std')
    )
    
    # Add 'All' row
    all_summary = pd.DataFrame({
        'species_count': [files_metadata['species_eng'].nunique()],
        'sample_count': [files_metadata['file_name'].size],
        'length_mean': [files_metadata['length'].mean()],
        'length_std': [files_metadata['length'].std()]
    }, index=['Total'])
    
    summary_table = pd.concat([dataset_summary, all_summary])

    # Plot table
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.axis('tight')
    ax3.axis('off')
    table = ax3.table(cellText=summary_table.values, colLabels=summary_table.columns, rowLabels=summary_table.index, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.show()

    return files_metadata, summary_table

#%% visual spec:  
def visual_spec(input_data_path, sr=48000, hop_length=256, n_fft=1024, n_mels=100, min_freq=200, max_freq=15000):
    """
    Visualizes the waveform, spectrogram, and mel-spectrogram of the first .wav file in the input_data_path.
    
    Parameters:
    - input_data_path: Path where .wav files are stored.
    - sr: Sample rate for loading the audio file.
    - hop_length: Number of audio frames between STFT columns.
    - n_fft: Length of the FFT window.
    - n_mels: Number of mel bands to generate.
    - min_freq: Minimum frequency for mel filter bank.
    - max_freq: Maximum frequency for mel filter bank.
    
    Usage:
    visual_spec(input_data_path, sr=48000, hop_length=256, n_fft=1024, n_mels=100, min_freq=200, max_freq=15000)
    """
    
    # Change to the directory where wav files are located
    os.chdir(input_data_path)
    wav_files = glob.glob("*.wav")  # make list of all wav files
    if len(wav_files) == 0:
        print("No .wav files found in the directory.")
        return

    file = wav_files[1]  # Pick the first file
    signal, sr = librosa.load(file, sr=sr)
    
    # Spectrogram
    S = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=n_fft))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Mel-Spectrogram
    S_mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=min_freq, fmax=max_freq)
    S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
    
    # Calculate time and frequency bins
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    times_mel = librosa.frames_to_time(np.arange(S_db_mel.shape[1]), sr=sr, hop_length=hop_length)

    # Create figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    # Plot spectrogram
    img1 = ax[0].contourf(times, librosa.fft_frequencies(sr=sr, n_fft=n_fft), S_db, cmap='viridis', levels=np.linspace(S_db.min(), S_db.max(), 100))
    ax[0].set_title('Spectrogram (dB)')
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    ax[0].set_ylim(0, max_freq) 
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_xlabel('Time (sec)')
    ax[0].minorticks_on()
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))

    # Plot mel-spectrogram with the number of filters on the y-axis
    img2 = ax[1].imshow(S_db_mel, aspect='auto', origin='lower', cmap='viridis', extent=[times_mel.min(), times_mel.max(), 0, n_mels])
    ax[1].set_title('Mel-Spectrogram (dB)')
    ax[1].set_ylabel('Mel Filter Index')
    ax[1].set_xlabel('Time (sec)')
    fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')
    ax[1].minorticks_on()
    ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))

    # Adjust layout    
    plt.subplots_adjust(0.8)
    plt.tight_layout()
    plt.show() 
  
#%% visual augmentations
def plot_spectrogram(x, sr, ax, title):
    """
    Plot the spectrogram of an audio signal.

    Parameters:
    x : array-like
        Audio signal.
    sr : int
        Sample rate of the audio signal.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    title : str
        Title of the plot.
    """
    # Compute the spectrogram
    spectrogram = np.abs(librosa.core.stft(x, hop_length=256, n_fft=1024))
    log_spec = librosa.amplitude_to_db(spectrogram)
    
    # Calculate time and frequency bins
    times = librosa.frames_to_time(np.arange(log_spec.shape[1]), sr=sr, hop_length=256)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    
    # Plot the spectrogram
    cax = ax.contourf(times, freqs, log_spec, cmap='viridis', levels=np.linspace(log_spec.min(), log_spec.max(), 100))
    cbar = plt.colorbar(cax, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)')
    ax.set_title(title)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    
    # Limit y-axis to max_freq
    ax.set_ylim(0, 15000)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
def augment_and_display(file_path, file_name):
    """
    Apply different augmentations to an audio file and display their spectrograms.

    Parameters:
    file_path : str
        Path to the audio file.
    file_name : str
        Name of the audio file.
    
    Usage:
    
    """
    # Load the audio signal
    signal, sr = librosa.load(f"{file_path}/{file_name}", sr=None)
    
    # Define augmentations
    augmentations = {
        'Original': signal,
        'Gaussian Noise': AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1.0)(samples=signal, sample_rate=sr),
        'Time Stretch': TimeStretch(min_rate=0.7, max_rate=1.3, p=1.0)(samples=signal, sample_rate=sr),
        'Pitch Shift': PitchShift(min_semitones=-3, max_semitones=3, p=1.0)(samples=signal, sample_rate=sr),
    }
    
    # Define combined augmentation
    combined_augmentation = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=1),
        TimeStretch(min_rate=0.7, max_rate=1.3, p=1),
        PitchShift(min_semitones=-3, max_semitones=3, p=1)
    ])(samples=signal, sample_rate=sr)
    
    # Add combined augmentation to the dictionary
    augmentations['Combined Augmentation'] = combined_augmentation

    # Plot spectrograms
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))  # Changed to 2x3 grid to fit 5 plots
    axes = axes.flatten()

    for ax, (title, augmented_signal) in zip(axes, augmentations.items()):
        plot_spectrogram(augmented_signal, sr, ax, title=title)

    plt.tight_layout()
    plt.show()
    
# Example usage
# file_path = r'C:\Users\ATC Technologies\Desktop\drive-download-20240827T124404Z-001'
# file_name = r'דגימה 6 שניות.wav'
# augment_and_display(file_path, file_name)


# # Example usage
# file_path = r'G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings\Agmon'
# file_name = r'518_32_S4A13411_20210610_120000.wav'
# augment_and_display(file_path, file_name)

#%% display segmentations
def display_segment(input_data_path, segment_duration=3, sr=48000, overlap=0, method='wgn', factor=30, hop_length=256, n_fft=1024):
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
        
        Usage:
        display_segment(input_data_path, segment_duration=3, sr=48000, overlap=0, method='wgn', factor=30, hop_length=256, n_fft=1024)
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
    
    os.chdir(input_data_path)
    wav_files = glob.glob("*.wav")  # Make list of all wav files
    if len(wav_files) == 0:
        print("No .wav files found in the directory.")
        
    file = wav_files[6]  # Pick the first file
    signal, sr = librosa.load(file, sr=sr)
    segment_length = segment_duration * sr
    step = int(segment_length * (1 - overlap))
    num_samples = len(signal)
    segmented_data_list = []
    
    # Segment the signal
    for start in range(0, num_samples, step):
        end = start + segment_length
        if end > num_samples:
            segment = seg_padding(signal[start:], segment_length, method=method, factor=factor)
        else:
            segment = signal[start:end]
        segmented_data_list.append(segment)
        
        # Limit the number of segments to 3 for plotting
        if len(segmented_data_list) >= 3:
            break

    # Plotting the spectrogram of the full signal
    S_full = np.abs(librosa.stft(signal, hop_length=hop_length, n_fft=n_fft))
    S_db_full = librosa.amplitude_to_db(S_full, ref=np.max)
    
    # Creating a GridSpec for flexible layout
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 4, height_ratios=[2, 1], width_ratios=[1, 1, 1, 0.05], figure=fig)  # One large top row, three smaller plots in a single row
    
    # Full signal spectrogram in the top row (spanning all 3 columns)
    ax_full = fig.add_subplot(gs[0, :3])
    img1 = ax_full.imshow(S_db_full, aspect='auto', cmap='viridis', origin='lower', 
                          extent=[0, len(signal)/sr, 0, sr/2])
    fig.colorbar(img1, ax=ax_full, format='%+2.0f dB')
    ax_full.set_title('Full Signal Spectrogram')
    ax_full.set_ylabel('Frequency (Hz)')
    ax_full.set_xlabel('Time (sec)')
    ax_full.set_ylim(0, 15000)
    
    # Plotting the segments in the second row, side by side
    for i, segment in enumerate(segmented_data_list):
        S_segment = np.abs(librosa.stft(segment, hop_length=hop_length, n_fft=n_fft))
        S_db_segment = librosa.amplitude_to_db(S_segment, ref=np.max)

        ax_segment = fig.add_subplot(gs[1, i])  # Columns 0, 1, 2 of the second row
        img2 = ax_segment.imshow(S_db_segment, aspect='auto', cmap='viridis', origin='lower',
                                 extent=[0, len(segment)/sr, 0, sr/2])
        # fig.colorbar(img2, ax=ax_segment, format='%+2.0f dB')
        ax_segment.set_title(f'Segment {i+1} Spectrogram')
        ax_segment.set_ylabel('Frequency (Hz)')
        ax_segment.set_xlabel('Time (sec)')
        ax_segment.set_ylim(0, 15000)
    
    plt.tight_layout()
    plt.show() 
    
  


#%% more

def plot_activation():
    # Define the activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum(axis=0)
    
    # Generate input data
    x = np.linspace(-10, 10, 400)
    
    # Prepare subplots
    plt.figure(figsize=(10, 8))
    
    # Plot Sigmoid
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid(x), label="Sigmoid", color="b")
    plt.title("Sigmoid")
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.grid(True)
    
    # Plot Tanh
    plt.subplot(2, 2, 2)
    plt.plot(x, tanh(x), label="Tanh", color="g")
    plt.title("Tanh")
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.grid(True)
    
    # Plot ReLU
    plt.subplot(2, 2, 3)
    plt.plot(x, relu(x), label="ReLU", color="r")
    plt.title("ReLU")
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.grid(True)
    
    # Plot Softmax
    plt.subplot(2, 2, 4)
    x_softmax = np.array([x, x + 1, x + 2])  # Softmax applied to a vector of values
    plt.plot(x, softmax(x_softmax)[0], label="Softmax", color="m")  # Plot only one row of the softmax output
    plt.title("Softmax")
    plt.xlabel("Input (x)")
    plt.ylabel("Output (Probability)")
    plt.grid(True)
    
    # Show the plots with adjusted layout
    plt.tight_layout()
    plt.show()
    
    
#########################################################################

def get_wav_duration(wav_file_path):
    """Helper function to get the duration of a wav file using pydub."""
    try:
        audio = AudioSegment.from_wav(wav_file_path)
        duration = len(audio) / 1000.0  # Duration in seconds
        return duration
    except Exception as e:
        print(f"Error processing {wav_file_path}: {e}")
        return None  # Return None for files that can't be processed

def process_wav_files(path, use_threading=False, max_workers=4):
    """
    Process all .wav files in subfolders of the given path, 
    and generate a histogram of file lengths and a table with 
    the number of files per dataset (folder). Uses multiprocessing.

    Args:
        path (str): Path to the main directory containing dataset folders.
        use_threading (bool): If True, use ThreadPoolExecutor instead of ProcessPoolExecutor.
        max_workers (int): Maximum number of worker threads/processes to use.
    
    Returns:
        pd.DataFrame: Table with number of .wav files per dataset.
    """
    dataset_lengths = {}
    file_paths = []
    
    # First pass to gather all .wav file paths and dataset names
    for root, dirs, files in os.walk(path):
        for dataset in dirs:
            dataset_path = os.path.join(root, dataset)
            wav_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]
            dataset_lengths[dataset] = len(wav_files)
            file_paths.extend(wav_files)
    
    total_files = len(file_paths)
    
    # Choose executor based on user choice
    Executor = ThreadPoolExecutor if use_threading else ProcessPoolExecutor
    file_lengths = []
    
    with tqdm(total=total_files, desc="Processing .wav files", unit="file") as pbar:
        with Executor(max_workers=max_workers) as executor:
            futures = [executor.submit(get_wav_duration, file) for file in file_paths]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        file_lengths.append(result)
                except Exception as e:
                    print(f"Error in future: {e}")
                pbar.update(1)  # Update progress bar after each file finishes
    
    # Check if file_lengths is empty before plotting
    if file_lengths:
        # Plot histogram of file lengths
        plt.figure(figsize=(10, 6))
        plt.hist(file_lengths, bins=30, color='blue', edgecolor='black')
        plt.title('Distribution of .wav File Lengths')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of Files')
        plt.xlim(0, max(file_lengths) + 1)  # Set x-axis limit for better visibility
        plt.grid(True)
        plt.show()
    else:
        print("No valid .wav files processed. Histogram will not be plotted.")

    # Create a summary table
    summary_df = pd.DataFrame(list(dataset_lengths.items()), columns=['Dataset', 'Number of Files'])
    print(summary_df)
    
    return summary_df



# summary_table = process_wav_files(r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\Recordings (new)\DeepBirdRecordings")

####################################################################


# def plot_species_table(metadata, title='Species Count Table'):
#     # Extract species names and their counts from the metadata dictionary
#     species_data = metadata['datasets']['Agmon']
    
#     # Convert the species data into a pandas DataFrame
#     df = pd.DataFrame({
#         'Species': list(species_data.keys()),
#         'Count': list(species_data.values())
#     })
    
#     # Sort data for cleaner display (optional)
#     df = df.sort_values(by='Species')

#     # Create a matplotlib figure
#     fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

#     # Hide the axes
#     ax.axis('off')
#     ax.axis('tight')

#     # Create a styled pandas table with better formatting
#     styled_table = df.style.set_table_styles(
#         [{'selector': 'thead th', 'props': [('background-color', 'lightgrey'),
#                                             ('color', 'black'), 
#                                             ('font-weight', 'bold')]},
#          {'selector': 'tbody td', 'props': [('border', '1px solid black')]}]
#     ).background_gradient(subset=['Count'], cmap="Blues").render()

#     # Use plt to draw the table
#     ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

#     # Add a title
#     plt.title(title, fontsize=16, fontweight='bold', pad=20)

#     # Show the plot
#     plt.tight_layout()
#     plt.show()

# # Example usage with the `files_metadata` dictionary
# plot_species_table(files_metadata, title="Species Count in Agmon Dataset")

###############################################################################
# plot mel scale function
# import numpy as np
# import matplotlib.pyplot as plt

# # Function to convert frequency (Hz) to Mel scale
# def hz_to_mel(frequency):
#     return 2595 * np.log10(1 + frequency / 700)

# # Function to convert Mel scale back to frequency (Hz)
# def mel_to_hz(mel):
#     return 700 * (10**(mel / 2595) - 1)

# # Generate frequency values (0 to 8000 Hz, typical audio range)
# frequencies = np.linspace(0, 20000, 1000)

# # Convert frequencies to Mel scale
# mel_values = hz_to_mel(frequencies)

# # Plotting
# plt.figure(figsize=(6, 8))
# plt.plot(mel_values, frequencies, label="Mel Scale")
# plt.title("Mel Scale vs Frequency")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Mel Scale")
# plt.grid(True)
# plt.legend()
# plt.show()


###############################################################################
# plot mel filters(Mel Filter Bank)

# # Function to convert frequency (Hz) to Mel scale
# def hz_to_mel(frequency):
#     return 2595 * np.log10(1 + frequency / 700)

# # Function to convert Mel scale back to frequency (Hz)
# def mel_to_hz(mel):
#     return 700 * (10**(mel / 2595) - 1)

# # Function to create Mel filter banks directly in the Mel scale
# def mel_filter_banks(num_filters, NFFT, sample_rate, low_freq=0, high_freq=None):
#     if high_freq is None:
#         high_freq = sample_rate / 2  # Nyquist frequency

#     # Convert low and high frequencies to Mels
#     low_mel = hz_to_mel(low_freq)
#     high_mel = hz_to_mel(high_freq)

#     # Generate equally spaced Mel points
#     mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
#     # Convert Mel points back to Hz
#     hz_points = mel_to_hz(mel_points)
    
#     # Convert Hz points to FFT bin numbers
#     bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)
    
#     # Initialize filter bank array
#     filter_banks = np.zeros((num_filters, int(NFFT / 2 + 1)))

#     # Create triangular filters on the Mel scale
#     for i in range(1, num_filters + 1):
#         left = bin_points[i - 1]
#         center = bin_points[i]
#         right = bin_points[i + 1]

#         # Rising edge of the triangle
#         for j in range(left, center):
#             filter_banks[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        
#         # Falling edge of the triangle
#         for j in range(center, right):
#             filter_banks[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])

#     return filter_banks, hz_points

# # Parameters
# sample_rate = 16000  # Sampling frequency
# NFFT = 512           # Number of FFT points
# num_filters = 10     # Number of Mel filters

# # Generate Mel filter banks
# filters, hz_points = mel_filter_banks(num_filters, NFFT, sample_rate)

# # Generate the Mel scale for amplitude scaling
# frequencies = np.linspace(0, sample_rate / 2, int(NFFT / 2 + 1))  # Frequency range from 0 to Nyquist
# mel_scale = hz_to_mel(frequencies)  # Convert frequencies to Mel scale
# mel_normalized = mel_scale / mel_scale.max()  # Normalize Mel scale for amplitude scaling

# # Scale filter banks' amplitude to stay within the Mel scale
# scaled_filters = filters * mel_normalized

# # Plot the Mel filter banks with scaled amplitude
# plt.figure(figsize=(6, 8))
# for i in range(num_filters):
#     plt.plot(scaled_filters[i], frequencies)

# # Adjust the plot appearance
# plt.title("Mel Filter Banks with Amplitude Limited by Mel Scale (Frequency on Y-axis)")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Amplitude")
# plt.grid(True)
# plt.show()



##########################################

# # Function to convert frequency (Hz) to Mel scale
# def hz_to_mel(frequency):
#     return 2595 * np.log10(1 + frequency / 700)

# # Function to convert Mel scale back to frequency (Hz)
# def mel_to_hz(mel):
#     return 700 * (10**(mel / 2595) - 1)

# # Function to create Mel filter banks
# def mel_filter_banks(num_filters, NFFT, sample_rate, low_freq=0, high_freq=None):
#     if high_freq is None:
#         high_freq = sample_rate / 2  # Nyquist frequency

#     # Convert low and high frequencies to Mels
#     low_mel = hz_to_mel(low_freq)
#     high_mel = hz_to_mel(high_freq)

#     # Generate equally spaced Mel points
#     mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
#     # Convert Mel points back to Hz
#     hz_points = mel_to_hz(mel_points)
    
#     # Convert Hz points to FFT bin numbers
#     bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)
    
#     # Initialize filter bank array
#     filter_banks = np.zeros((num_filters, int(NFFT / 2 + 1)))

#     # Create triangular filters
#     for i in range(1, num_filters + 1):
#         left = bin_points[i - 1]
#         center = bin_points[i]
#         right = bin_points[i + 1]

#         # Rising edge of the triangle
#         for j in range(left, center):
#             filter_banks[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        
#         # Falling edge of the triangle
#         for j in range(center, right):
#             filter_banks[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])

#     return filter_banks

# # Parameters
# sample_rate = 48000  # Sampling frequency
# NFFT = 1024           # Number of FFT points
# num_filters = 15     # Number of Mel filters

# # Generate Mel filter banks
# filters = mel_filter_banks(num_filters, NFFT, sample_rate)

# # Plot the Mel filter banks with swapped axes
# plt.figure(figsize=(6, 8))
# for i in range(num_filters):
#     plt.plot(filters[i], np.linspace(0, sample_rate / 2, int(NFFT / 2 + 1)), color='black')

# # Set axis labels and titles
# plt.title("Mel filters")
# plt.ylabel("Frequency (Hz)")
# plt.xlabel("Amplitude")
# plt.xlim(0, 1)  # Set x-axis limit for amplitude
# plt.ylim(0, sample_rate / 2)  # Set y-axis limit for frequency
# plt.grid(True)
# plt.show()