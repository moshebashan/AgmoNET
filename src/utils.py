# -*- coding: utf-8 -*-
"""


@author: 
"""

import os
import pandas as pd
import pickle
import librosa, librosa.display
import logging
import soundfile as sf

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


#%% save milon.txt as .pkl file:
milon_path = r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\agmonet\resources"
milon_txt_path = os.path.join(milon_path, 'milon.txt')
milon = pd.read_csv(milon_txt_path, sep="\t", header=None, encoding=('iso8859_8'))
milon = milon.dropna(axis=1, how='all')
pickle_path = os.path.join(milon_path, 'milon.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump(milon, f)
print(f"milon saved as pickle at: {pickle_path}")

#%% play sound:
def play_sound(file=r"G:\האחסון שלי\לימודים\תואר שני\project bird recognition\python\AgmoNET\resources\notification_sound.wav"):
    import simpleaudio as sa 
    sa.WaveObject.from_wave_file(file).play()
    
    
    
# # play sound from WAV file:
# from playsound import playsound
# playsound('D:/google drive/לימודים/תואר שני/project bird recognition/Recordings (new)/samples/blackbird.wav')

# # play sound from 2D array:
# import sounddevice as sd
# sd.play(y)

############################################################################
#%% read_metadata:
def read_metadata(input_data_path, milon_path, sr=48000):
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", header=None, encoding='iso8859_8')
    milon_dict = dict(zip(milon[0], milon[3]))  # key: label_id, value: species_eng
    files_metadata_list = []
    # files_metadata = pd.DataFrame(columns=['dataset', 'file_name', 'species_eng' , 'length'])
    # dataset_info = {}
    
    print('INFO: Reading files info')
    for root, dirs, files in os.walk(input_data_path):
        # Skip directories without .wav files
        wav_files = [file for file in files if file.endswith(".wav")]
        if not wav_files:
            continue
        dataset_name = os.path.basename(root)
    
        for f, file in enumerate(wav_files, 1):
            total_files = len(wav_files)
            print(f'\rINFO: Processing - Dataset: {dataset_name} File: {f}/{total_files}', end='', flush=True)
            file_path = os.path.join(root, file)            
            with sf.SoundFile(file_path) as audio_file:
                length = len(audio_file) / audio_file.samplerate
            
            
            # signal, sr = librosa.load(file_path, sr=sr)
            # length = len(signal) / sr
            # size = os.path.getsize(file_path)
    
            label_id = file.split('_')[0] if '_' in file else file[:3]  # Adapt this if the index extraction logic varies
            species_eng = milon_dict.get(label_id, 'Unknown')
            # species_eng = milon[milon[0] == label_id][3].to_string(index=False).strip()
            files_metadata_list.append([dataset_name, file, species_eng, length])
            # files_metadata.loc[len(files_metadata)] = [dataset_name, file, species_eng, length]
            
    files_metadata = pd.DataFrame(files_metadata_list, columns=['dataset', 'file_name', 'species_eng', 'length'])

    return files_metadata


#%% more            
############################################################################            
# calculate f1 score:
def calculate_f1(recall,precision):
    '''
    Usage:
        f1 = calculate_f1()
    '''
    f1 = 2*((recall*precision)/(recall+precision))
    print(f'F1 score: {round(f1,2)}')
    return f1            
            
        
################################################################################

'''
EcoNameTranslator:
https://pypi.org/project/EcoNameTranslator/
'''
# from EcoNameTranslator import to_common
# common_names = to_common(['Turdus merula'])
# print(common_names)

 #####################################################
# def save_species_data_to_excel(metadata, output_file='species_count.xlsx'):
#     # Extract species names and their counts from the metadata dictionary
#     species_data = metadata['datasets']['Agmon']
    
#     # Convert the species data into a pandas DataFrame
#     df = pd.DataFrame({
#         'Species': list(species_data.keys()),
#         'Count': list(species_data.values())
#     })
#     os.chdir(root_path)
#     # Save the DataFrame to an Excel file
#     df.to_excel(output_file, index=False)

#     print(f"Data successfully saved to {output_file}")

# # Example usage with the `files_metadata` dictionary
# save_species_data_to_excel(files_metadata, 'species_count.xlsx') 

# ######################################################################

# # Assuming files_metadata['details'] is your dataframe
# df = files_metadata['details']

# # Plot 1: Histogram for the 'length' column
# plt.figure(figsize=(8, 6))
# plt.hist(df['length'], bins=50, color='blue', edgecolor='black')
# plt.title('Histogram of Length')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Plot 2: Histogram for the 'length' column where dataset is 'Agmon'
# agmon_df = df[df['dataset'] == 'Agmon']

# plt.figure(figsize=(8, 6))
# plt.hist(agmon_df['length'], bins=80, color='blue', edgecolor='black')
# plt.title('Histogram of sample length of Agmon dataset')
# plt.xlabel('Length [sec]')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()        
