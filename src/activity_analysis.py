# -*- coding: utf-8 -*-
"""
Species activity analysis

@author: Moshe Bashan
"""

#%% import packages
# IMPORT PACKAGES:
import glob, os
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.dates as mdates
from matplotlib import cm, colors
from matplotlib.lines import Line2D
import heapq
from dateutil.relativedelta import relativedelta
import statistics
import math
from scipy import stats
from scipy.stats import ttest_ind
from astral import LocationInfo
from astral.sun import sun
from datetime import date, datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytz import timezone

 

#%% Species_activity_aggregator (peilut species)
def species_activity_aggregator(milon_path,df2_path,species_list,threshold=0.1):
    """
       Processes and aggregates species activity data from Excel files and saves the results into new Excel files.
    
       Parameters:
       - milon_path (str): Path to the directory containing the 'milon.txt' file, which maps species names to their IDs.
       - df2_path (str): Path to the directory containing the input Excel files with species activity data.
       - species_list (list of str or int): List of species names or IDs to be processed.
       - threshold (float): Threshold value to filter species activity data based on the median probability. Default is 0.1.
    
       Returns:
       - dict: A dictionary where keys are species names and values are DataFrames containing aggregated activity data.
    
       Note:
       - The function assumes that the filenames of the input Excel files contain date and hour information, and that the data columns include 'species_id', 'prob1_median', 'rank1_counts', etc.
       - The output files are named according to the species label and station number.
       - The date range is currently hardcoded and needs adjustment to dynamically reflect the date range from the Excel files.
    
       Usage:
           species_activity_data = species_activity_aggregator(milon_path, df2_path, species_list, threshold)
    """

    # Read species mapping
    milon = pd.read_csv(os.path.join(milon_path, "milon.txt"), sep="\t", encoding='iso8859_8')
    
    species_id_list = []
    species_eng_list = []
    # Look up species IDs and names in milon
    for species in species_list:
        matched_rows = milon[milon.apply(lambda row: row.astype(str).str.contains(str(species), case=False, na=False).any(), axis=1)]
        if not matched_rows.empty:
            matched_row = matched_rows.iloc[0]
            species_id_list.append(int(matched_row['species id']))
            species_eng_list.append(matched_row.get('species eng', 'Unknown'))
        else:
            species_id_list.append(None)
            species_eng_list.append('Unknown')
            print(f"Warning: Species '{species}' not found in the milon data.")
          
    # Gather Excel files
    input_file_list = glob.glob(os.path.join(df2_path, "*.xlsx"))
    
    # Date range ***this part need to fixed to take the dates from the xlsx files instead of hardcoding it
    day_start = datetime.strptime('2020-11-18', '%Y-%m-%d')
    day_end = datetime.strptime('2022-11-22', '%Y-%m-%d')
    date_list_full = [(day_start + timedelta(n)).date() for n in range((day_end - day_start).days + 1)]
    
    # Determine station from filename
    station = None
    if input_file_list:
        if '411' in input_file_list[0]:
            station = '2'
        elif '335' in input_file_list[0]:
            station = '3'
    
    # Initialize lists
    files_list = []
    files_names = []
    empty_row = [0] * 24
    
    # Read input files
    for index, file_name in enumerate(input_file_list):
        print(f'\rReading file {index + 1}/{len(input_file_list)}', end='')
        file_df = pd.read_excel(file_name)
        files_names.append(os.path.basename(file_name))
        files_list.append(file_df)
    
    species_activity_data = {}        
    # Process each species
    for s, species_eng in enumerate(species_eng_list):
        species_id = species_id_list[s]
        # Create DataFrame with dates as index and hours as columns, initializing all values to 0
        activity_df = pd.DataFrame(0, index=date_list_full, columns=[f"{i:02}" for i in range(24)])
        # Process each file   
        for file_df, file_name in zip(files_list, files_names):
            day = datetime.strptime(file_name[9:13]+'-'+file_name[13:15]+'-'+file_name[15:17], '%Y-%m-%d').date()            
            hour = file_name[18:20]            
            species_row = file_df[file_df['species_id'] == species_id]
            count = 0
            if not species_row.empty:
                if (species_row['prob1_median'] >= threshold).any():
                    count += species_row['rank1_counts'].sum()
                if (species_row['prob2_median'] >= threshold).any():
                    count += species_row['rank2_counts'].sum()
                if (species_row['prob3_median'] >= threshold).any():
                    count += species_row['rank3_counts'].sum()
            activity_df.at[day, hour] = count                       
        
        # Save results in .xlsx files
        output_path = os.path.join(df2_path, 'species_activity_data')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        output_file_name = f'activity_{species_eng.replace(" ", "_")}_{station}.xlsx'
        activity_df.to_excel(os.path.join(output_path, output_file_name))
        print(f'File: {output_file_name} Successfully saved')
        species_activity_data['species_eng'] = activity_df
    return species_activity_data


#%% Handle_missing_data (rem_missing_times):
def handle_missing_data(species_activity_data_path):
    ''' 
    Processes species activity Excel files to identify and handle missing data records by overwriting original files with NaN values for dates with missing data.
    This function identifies dates with zero activity across multiple species at each station and treats these as missing data points.
    It assumes that if data from at least 10 species is available, any day with zero activity across all species indicates missing data rather than natural inactivity.
    The function creates a DataFrame named 'missing_data.xlsx', which contains the dates with zero activity for each station.
    This DataFrame is saved in the specified directory. Additionally, the function updates the original activity Excel files, replacing data with NaN for the identified missing dates.
    
    Parameters:
        species_activity_data_path (str): Path to the directory containing activity Excel files. Files should be named with the pattern 'activity_{species}_{station}.xlsx'.
        
    Returns:
        missing_data (DataFrame): A DataFrame containing the dates with zero activity (missing data) for each station.
    
    Usage:
        missing_data = handle_missing_data(species_activity_data_path)
    
    Notes:
    - The function expects a sufficient number of files (at least 10) per station to accurately determine missing data points. If fewer than 10 files are found for a station, the function will not perform the missing data calculation for that station.
    - Dates with missing data are determined by summing the counts of activity across all hours and species; if the total is zero, the date is flagged as missing.
    
    '''
    files = [f for f in os.listdir(species_activity_data_path) if f.startswith('activity') and f.endswith('.xlsx')]
    station_files = {}    
    for file in files:
        station = file.split('_')[-1].split('.')[0]
        station_files.setdefault(station, []).append(file)   
    zero_sum_dates_dict = {}    
    for station, file_list in station_files.items():
        zero_sum_dates_dict[station] = []
        if len(file_list) < 10:           
            print(f"Can't calculate missing times for Station {station}, use at least 10 files per station")
            exit
        combined_df = pd.concat([pd.read_excel(os.path.join(species_activity_data_path, f)).rename(columns={"Unnamed: 0": "date"}) for f in file_list])
        combined_df = combined_df.groupby('date').sum().reset_index()
        zero_sum_dates_dict[station] = combined_df[combined_df.iloc[:, 1:].sum(axis=1) == 0]['date'].tolist()    
    max_rows = max(len(dates) for dates in zero_sum_dates_dict.values())
    missing_data = pd.DataFrame({station: dates + [np.nan] * (max_rows - len(dates))
        for station, dates in zero_sum_dates_dict.items()})    
    # Save the missing_data DataFrame
    missing_data.to_excel(os.path.join(species_activity_data_path, 'missing_data.xlsx'), index=False)   
    # Replace values in original files with NaN where sums of rows are zero:
    for station, file_list in station_files.items():
        zero_dates = zero_sum_dates_dict.get(station, [])
        for file in file_list:
            file_path = os.path.join(species_activity_data_path, file)
            df = pd.read_excel(file_path)
            df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
            df.set_index('date', inplace=True)                        
            df.loc[df.index.isin(zero_dates), :] = np.nan # Replace values with NaN where date is in zero_dates          
            df.reset_index(inplace=True)
            df.to_excel(file_path, index=False) #overwrite the original file
    print('All files are updated and mising data file was saved')
    return missing_data

#%% Get_sun_times:
def get_sun_times(date_start,date_end, dst_transition=False):
    '''
    calculates sunrise and sunset times for each day within a specified date range for Jerusalem, Israel.

    Parameters:
        date_start (str or datetime.date): Start date for retrieving sun times in "MM-DD-YYYY" format or as a datetime.date object.
        date_end (str or datetime.date): End date for retrieving sun times in "MM-DD-YYYY" format or as a datetime.date object.

    Returns:
        pd.DataFrame: DataFrame with columns 'date', 'sunrise', and 'sunset', where each row represents a day in the specified range.

    Usage:
        sun_times = get_sun_times(date_start, date_end, dst_transition=False)  
    '''

    sun_times = pd.DataFrame({},columns=['date','sunrise','sunset'])
    location = LocationInfo("Jerusalem", "Israel", "Asia/Jerusalem",  33.10, 35.61)
    sunrise = []
    sunset = []
    # sun_times1 = {}
    date_list = []
    day_start = datetime.strptime(date_start, '%m-%d-%Y')
    day_end = datetime.strptime(date_end, '%m-%d-%Y')
    for n in range(int((day_end - day_start).days)+1):
        d = day_start + timedelta(n)
        date_list.append(d.date())   
    for day in range(len(date_list)):
        s = sun(location.observer, date=date_list[day], tzinfo=location.timezone)
        # sun_times1[date_list[day].strftime('%m-%d-%Y')] = s
        sunrise.append(s['sunrise'].strftime('%H:%M'))
        sunset.append(s['sunset'].strftime('%H:%M'))
    sun_times['date'] = date_list
    sun_times['sunrise'] = sunrise
    sun_times['sunset'] = sunset 
    
    # If dst_transition is True, check and adjust times
    if not dst_transition:
        for i in range(1, len(sun_times)):
            # Convert the sunrise and sunset times to datetime objects
            prev_sunrise = datetime.strptime(sun_times.loc[i-1, 'sunrise'], '%H:%M')
            curr_sunrise = datetime.strptime(sun_times.loc[i, 'sunrise'], '%H:%M')
            prev_sunset = datetime.strptime(sun_times.loc[i-1, 'sunset'], '%H:%M')
            curr_sunset = datetime.strptime(sun_times.loc[i, 'sunset'], '%H:%M')
            
            # Calculate differences in sunrise and sunset times in minutes
            sunrise_diff = (curr_sunrise - prev_sunrise).total_seconds() / 60  # difference in minutes
            sunset_diff = (curr_sunset - prev_sunset).total_seconds() / 60  # difference in minutes
            
            # If either sunrise or sunset difference exceeds 30 minutes, adjust both times
            if abs(sunrise_diff) > 30 or abs(sunset_diff) > 30:
                # Adjust by adding or subtracting one hour
                if sunrise_diff < 30 or sunset_diff < 30:
                    # Add one hour to both times
                    adjusted_sunrise = curr_sunrise + timedelta(hours=1)
                    adjusted_sunset = curr_sunset + timedelta(hours=1)
                else:
                    # Subtract one hour from both times
                    adjusted_sunrise = curr_sunrise - timedelta(hours=1)
                    adjusted_sunset = curr_sunset - timedelta(hours=1)
                
                # Store the adjusted times back in the DataFrame in '%H:%M' format
                sun_times.loc[i, 'sunrise'] = adjusted_sunrise.strftime('%H:%M')
                sun_times.loc[i, 'sunset'] = adjusted_sunset.strftime('%H:%M')
    
    return sun_times
'''
DST transitions for 2020:
2020-03-27 to 2020-03-28: 2:00:00 -> 3:00:00
2020-10-25 to 2020-10-26: 3:00:00 -> 2:00:00

DST transitions for 2021:
2021-03-26 to 2021-03-27: 2:00:00 -> 3:00:00
2021-10-31 to 2021-11-01: 3:00:00 -> 2:00:00

DST transitions for 2022:
2022-03-25 to 2022-03-26: 2:00:00 -> 3:00:00
2022-10-30 to 2022-10-31: 3:00:00 -> 2:00:00
'''

#%% Activity_levels_comparison (peilut_compar):
def activity_levels_comparison(milon_path, species_activity_path, species_list, stations):
    '''
    Compare activity levels of specified species across multiple stations.
    This function reads species information and activity data,
    computes the mean and standard error of the mean (SEM) for each species,
    generates bar plots, performs statistical tests to compare activity levels,
    and marks significant differences with '*' on the plot next to the bar.

    Parameters:
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files.
        species_list (list of str or int): List of species names or IDs to compare.
        stations (list of str): List of station names to include in the comparison.

    Returns:
        None
    '''
    # Change directory to milon path and read species information
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding='iso8859_8')

    # Change directory to species activity path
    os.chdir(species_activity_path)

    # Create species list in English
    species_list_eng = []   
    for s in species_list:
        species_eng = milon[(milon == s).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)

    # Prepare dictionary to store data for each species across stations
    species_data = {}

    # Collect data for each station and species
    for station in stations:
        for species in species_list_eng:
            file_name = f'activity_{species}_{station}.xlsx'
            if os.path.exists(file_name):  # Check if the file exists
                peilut = pd.read_excel(file_name)
                peilut = peilut.rename(columns={"Unnamed: 0": "date"})
                data = peilut.iloc[:, 1:].values.flatten().tolist()

                if species not in species_data:
                    species_data[species] = {}
                species_data[species][station] = [value for value in data if not np.isnan(value)]

    # Initialize figure and axes for subplots
    fig, axes = plt.subplots(1, len(stations), figsize=(12, 6), sharey=True)

    # Prepare lists for means, SEMs, and labels
    overall_means = []
    overall_sems = []
    overall_labels = []

    # Collect mean, SEM, and labels for each species
    for species in species_list_eng:
        combined_means = []
        combined_sems = []
        has_data = True
        for station in stations:
            data = species_data[species].get(station, [])
            mean_value = np.nanmean(data) if data else np.nan
            combined_means.append(mean_value)
        
        if all(mean is not None for mean in combined_means):  # Check if all means are available
            overall_means.append(combined_means)
            overall_labels.append(species)
            # Calculate SEM for the combined data across both stations
            combined_data = []
            for station in stations:
                combined_data.extend(species_data[species].get(station, []))
            overall_sems.append(stats.sem(combined_data, nan_policy='omit'))  # SEM for the combined data

    # Flatten the overall_means to use for plotting
    flattened_means = [np.nanmean(m) for m in overall_means]
    sorted_indices = np.argsort(flattened_means)[::-1]
    
    # Sort means, labels based on sorted indices
    sorted_means = [flattened_means[i] for i in sorted_indices]
    sorted_labels = [overall_labels[i] for i in sorted_indices]
    sorted_sems = [overall_sems[i] for i in sorted_indices]  # Sort SEMs

    # Reverse the order for plotting
    sorted_means.reverse()
    sorted_labels.reverse()
    sorted_sems.reverse()  # Reverse SEMs for plotting

    # Plot the data for each station
    for station_index, station in enumerate(stations):
        ax = axes[station_index]
        y = np.arange(len(sorted_labels))
        means = [overall_means[i][station_index] for i in sorted_indices]

        # Reverse the means for the plotting
        means.reverse()
        sems = sorted_sems.copy()  # Use combined SEMs for error bars (already reversed)
        
        bars = ax.barh(y, means, alpha=1, ecolor='black', xerr=sems, capsize=5)  # Add error bars
        ax.set_xlabel('Calls per hour [Mean]')
        ax.set_title(f'Activity levels (station {station})')
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_labels)
        ax.invert_yaxis()  # Ensure the highest mean is at the top
        ax.xaxis.grid(True)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor grid lines

    # Statistical Testing and Marking Significant Differences
    for i, species in enumerate(sorted_labels):
        data_station_1 = species_data[species].get(stations[0], [])
        data_station_2 = species_data[species].get(stations[1], [])

        if data_station_1 and data_station_2:
            # Perform t-test between the two stations
            t_stat, p_value = ttest_ind(data_station_1, data_station_2, nan_policy='omit')

            if p_value < 0.001:
                # Determine which station has a higher mean
                mean_station_1 = np.nanmean(data_station_1)
                mean_station_2 = np.nanmean(data_station_2)

                if mean_station_1 > mean_station_2:
                    # Place the '*' next to the bar for station 1
                    ax = axes[0]
                    bar = ax.patches[i]
                    ax.text(
                        bar.get_width() + 0.1,  # Position to the right of the bar
                        bar.get_y() + bar.get_height() / 2,  # Vertically center the text
                        ' *',
                        ha='left',
                        va='center',
                        fontsize=13,
                        color='red'
                    )
                else:
                    # Place the '*' next to the bar for station 2
                    ax = axes[1]
                    bar = ax.patches[i]
                    ax.text(
                        bar.get_width() + 0.1,  # Position to the right of the bar
                        bar.get_y() + bar.get_height() / 2,  # Vertically center the text
                        ' *',
                        ha='left',
                        va='center',
                        fontsize=13,
                        color='red'
                    )

    plt.tight_layout()
    plt.show()

    # Print Mean and SEM for each species across both stations
    print("\nMean and SEM for each species:")
    for species in sorted_labels:
        data_combined = []
        for station in stations:
            data_combined.extend(species_data[species].get(station, []))

        mean_combined = np.nanmean(data_combined)
        sem_combined = stats.sem(data_combined, nan_policy='omit')

        print(f'{species}: Mean = {mean_combined:.4f}, SEM = {sem_combined:.4f}')

    # Print Statistical Test Results
    print("\nStatistical Test Results:")
    for species in sorted_labels:
        data_station_1 = species_data[species].get(stations[0], [])
        data_station_2 = species_data[species].get(stations[1], [])

        if data_station_1 and data_station_2:
            t_stat, p_value = ttest_ind(data_station_1, data_station_2, nan_policy='omit')
            result = 'Significant' if p_value < 0.001 else 'Not Significant'
            print(f'{species}: p-value = {p_value:.4f} ({result})')
        else:
            print(f'{species}: Insufficient data for comparison between stations.')
            
    
    
    
def activity_levels_comparison1(milon_path, species_activity_path, species_list, stations):
    '''
    Compare activity levels of specified species across multiple stations.
    This function reads species information and activity data,
    computes the mean and standard error of the mean (SEM) for each species,
    generates bar plots, performs statistical tests to compare activity levels,
    and marks significant differences with '*' on the plot next to the bar.

    Parameters:
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files.
        species_list (list of str or int): List of species names or IDs to compare.
        stations (list of str): List of station names to include in the comparison.

    Returns:
        None

    Usage:
        activity_levels_comparison(milon_path, species_activity_path, species_list, stations)
    '''
    # Change directory to milon path and read species information
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding='iso8859_8')
    
    # Change directory to species activity path
    os.chdir(species_activity_path)
    
    # Create species list in English
    species_list_eng = []   
    for s in range(len(species_list)):
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    
    # Initialize figure and axes for subplots
    fig, axes = plt.subplots(1, len(stations), figsize=(12, 6), sharey=True)
    
    # Prepare dictionaries to store data for statistical comparison
    station_data = {station: {} for station in stations}

    # Collect data for each station
    for stat_index, station in enumerate(stations):
        for i in range(len(species_list_eng)):
            species = species_list_eng[i]
            file_name = f'activity_{species}_{station}.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.rename(columns={"Unnamed: 0": "date"})
            data = peilut.iloc[:, 1:].values.flatten().tolist()
            station_data[station][species] = [value for value in data if not np.isnan(value)]  # Filter out NaN values

        # Calculate means and SEMs for the plot
        means = []
        sems = []
        labels = []
        for species, data in station_data[station].items():
            if data:  # Check if the filtered data list is not empty
                means.append(np.nanmean(data))  # Mean ignoring NaN
                sems.append(stats.sem(data, nan_policy='omit'))  # SEM ignoring NaN
                labels.append(species)
            else:
                means.append(np.nan)
                sems.append(np.nan)
                labels.append(species)

        # Plotting on the corresponding subplot
        y = np.arange(len(labels))  # The label locations
        ax = axes[stat_index]
        bars = ax.barh(y, means, xerr=sems, capsize=5, alpha=1, ecolor='black')
        ax.set_xlabel('Calls per hour [Mean]')
        ax.set_title(f'Activity levels (station {station})')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, va='baseline')
        ax.xaxis.grid(True)
        ax.invert_yaxis()
        # ax.set_xticks(range(0, round(np.nanmax(means) + 1)))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5)
    
    plt.tight_layout()

    # Statistical Testing and Marking Significant Differences
    for species in species_list_eng:
        data_station_1 = station_data[stations[0]].get(species, [])
        data_station_2 = station_data[stations[1]].get(species, [])
        
        if data_station_1 and data_station_2:
            # Perform statistical test: t-test
            t_stat, p_value = ttest_ind(data_station_1, data_station_2, nan_policy='omit')
            
            if p_value < 0.001:
                # Determine which station has a higher mean
                mean_station_1 = np.nanmean(data_station_1)
                mean_station_2 = np.nanmean(data_station_2)
                higher_station_index = 0 if mean_station_1 > mean_station_2 else 1
                
                # Position the asterisk next to the bar in the correct plot
                ax = axes[higher_station_index]
                bar_index = labels.index(species)  # Find the index of the bar for the species
                bar = ax.patches[bar_index]  # Get the specific bar
                ax.text(
                    bar.get_width() + 0.1,  # Position to the right of the bar
                    bar.get_y() + bar.get_height() / 2,  # Vertically center the text
                    ' *',
                    ha='left',
                    va='center',
                    fontsize=13,
                    color='red'
                )

    # Adding a legend for the significance indicator
    # significance_marker = Line2D([0], [0], marker='', color='w', label='* Significantly Higher',
    #                              markerfacecolor='red', markersize=12)
    # fig.legend(handles=[significance_marker],labels=['$*$ Significantly Higher'], loc='lower left', fontsize=10)

    plt.show()

    # Print Statistical Test Results
    print("Statistical Test Results:")
    for species in species_list_eng:
        data_station_1 = station_data[stations[0]].get(species, [])
        data_station_2 = station_data[stations[1]].get(species, [])
        
        if data_station_1 and data_station_2:
            t_stat, p_value = ttest_ind(data_station_1, data_station_2, nan_policy='omit')
            result = 'Significant' if p_value < 0.001 else 'Not Significant'
            print(f'{species}: p-value = {p_value:.4f} ({result})')
        else:
            print(f'{species}: Insufficient data for comparison between stations.')

#%% Activity_distribution_overview (peilut_total_summary):       
def activity_distribution_overview(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
    This function calculates and plots the mean daily vocal activity for multiple species over a specified time period.    
    The function reads species activity data from Excel files.
    Visualizations plot showing the vocal activity over time with imshow and highlights days with missing data.

    Parameters:
        species_list (list of str): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str or datetime.date): Start date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        date_end (str or datetime.date): End date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        tuple:
            - total_summary (pd.DataFrame): DataFrame containing the mean daily activity for each species.
    
    Usage:
        total_summary = activity_distribution_overview(species_list, milon_path, species_activity_path, 
            date_start, date_end, stations)
    '''
    
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding=('iso8859_8'))   
    if type(date_start) == str:
        date_start = datetime.strptime(date_start, "%m-%d-%Y").date()
    if type(date_end) == str:
        date_end = datetime.strptime(date_end, "%m-%d-%Y").date()
    dates = pd.date_range(date_start, date_end - timedelta(days=1), freq='d')  # list of dates
    species_list_eng = []
    for s in range(len(species_list)): 
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    
    activity_distribution = pd.DataFrame({}, index=dates, columns=species_list_eng) 
    os.chdir(species_activity_path)
    time_not_recorded = pd.read_excel('missing_data.xlsx')

    for station in stations:
        for i in range(len(species_list_eng)):
            species = str(species_list_eng[i])
            file_name = 'activity_' + species + '_' + station + '.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.rename(columns={"Unnamed: 0": "date"})
            activity_distribution[species] = peilut.iloc[:, 1:].mean(axis=1).to_frame().set_axis([species_eng], axis=1).set_index(peilut['date'])
        
        # Plotting:
        date_index = activity_distribution.index.date
        not_recorded = pd.to_datetime(time_not_recorded[station].dropna().tolist()).date
        not_recorded_pos = [np.where(date_index == date)[0][0] for date in not_recorded if date in date_index]
        
        fig, axs = plt.subplots(len(species_list_eng), 1)
        for j, species in enumerate(species_list_eng):
            max_v = activity_distribution[activity_distribution.columns[-1]].max()  # max value of the last species
            y = activity_distribution[species].values
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='black')

            img = axs[j].imshow(np.expand_dims(y, axis=1).T, aspect='20', cmap=cmap)
            axs[j].set_yticks([])
            axs[j].set_yticklabels([])
            
            if j < len(species_list_eng) - 1:
                axs[j].set_xticks([])  # Remove x-ticks for all subplots except the last one
            
            axs[j].grid(True, which='major', linestyle=':', linewidth='0.5', color='white')
            axs[j].set_ylabel(species_list_eng[j], rotation=0)
            axs[j].yaxis.set_label_coords(-0.45, 0)               
            
            # Add lines for missing data
            for pos in not_recorded_pos:  
                axs[j].axvline(pos, color='red', linestyle='-', linewidth=0.5, ymin=0, ymax=0.05)
        
        # Select first day of each month for the x-axis ticks (only for the last subplot)
        first_of_months = pd.date_range(start=date_start, end=date_end, freq='MS')
        month_labels = [d.strftime('%b %Y') for d in first_of_months]  # Format: 'Jan 2020'
        month_positions = [np.where(date_index == d.date())[0][0] for d in first_of_months if d.date() in date_index]
            
        axs[-1].set_xlabel('Time [Date]')
        axs[-1].set_xticks(month_positions)
        axs[-1].set_xticklabels(month_labels, rotation=90)
        
        fig.subplots_adjust(hspace=0.3)  # Adjust vertical space between subplots
        fig.suptitle(f'Vocal activity distribution over time\nStation {station}', fontsize=10, x=0.5)
        
        # Add colorbar and legend
        cbar = fig.colorbar(img, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label('Relative vocal activity\n Daily calls [mean]', labelpad=-20)
        cbar.set_ticks([0, max_v])
        cbar.set_ticklabels(['low', 'high'])
        
        red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label='Missing data')  
        axs[0].legend(handles=[red_line], loc=(-0.65, -33))
        
    return activity_distribution   
    
    
    
    
    
def activity_distribution_overview0(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
    This function calculates and plots the mean daily vocal activity for multiple species over a specified time period.    
    The function reads species activity data from Excel files.
    Visualizations plot showing the vocal activity over time with imshow and highlights days with missing data.

    Parameters:
        species_list (list of str): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str or datetime.date): Start date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        date_end (str or datetime.date): End date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        tuple:
            - total_summary (pd.DataFrame): DataFrame containing the mean daily activity for each species.
    
    Usage:
        total_summary = peilut_total_summary(species_list, milon_path, species_activity_path, 
            date_start, date_end, stations)
    
    Notes:
        - The function expects an Excel file named 'time not recorded.xlsx' in the specified directory.
        - Activity Excel files must be named according to the format 'activity_<species>_<station>.xlsx'.
        - The function generates and displays plots for the vocal activity overview but does not save them.
    '''
    
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding=('iso8859_8'))   
    if type(date_start) == str:
        date_start = datetime.strptime(date_start, "%m-%d-%Y").date()
    if type(date_end) == str:
        date_end = datetime.strptime(date_end, "%m-%d-%Y").date()
    dates = pd.date_range(date_start,date_end-timedelta(days=1),freq='d') #list of dates
    species_list_eng = []
    for s in range(len(species_list)): 
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    activity_distribution = pd.DataFrame({}, index=dates, columns=species_list_eng) 
    os.chdir(species_activity_path)
    time_not_recorded = pd.read_excel('missing_data.xlsx')
    for station in stations:
        for i in range(len(species_list_eng)):
            species = str(species_list_eng[i])
            file_name = 'activity_'+species+'_'+station+'.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.rename(columns={"Unnamed: 0": "date"})
            activity_distribution[species] = peilut.iloc[:, 1::].mean(axis=1).to_frame().set_axis([species_eng], axis=1).set_index(peilut['date'])
       
        #  Plotting:
        #x_pos = np.arange(total_summary.shape[0])
        date_index = activity_distribution.index.date
        not_recorded = pd.to_datetime(time_not_recorded[station].dropna().tolist()).date
        not_recorded_pos = [np.where(date_index == date)[0][0] for date in not_recorded if date in date_index] 
        fig, axs = plt.subplots(len(species_list_eng),1)
        for j, species in enumerate(species_list_eng):
            max_v = activity_distribution[activity_distribution.columns[len(species_list_eng)-1]].max() # the max value of the lest species
            y = activity_distribution[species].values
            cmap = plt.get_cmap('viridis')
            cmap.set_bad(color='black')
            #'YlGnBu'
            img = axs[j].imshow(np.expand_dims(y,axis=1).T,aspect='20',cmap=cmap)
            #img = axs[j].imshow(np.expand_dims(y,axis=1).T,aspect='30',vmin=0, vmax=total_summary[1::].max().max()*1.05)
            axs[j].set_yticks([])
            axs[j].set_yticklabels([])
            axs[j].set_xticks(range(0,len(date_index), 90))
            axs[j].set_xticklabels([])
            axs[j].grid(True, which='major', linestyle=':', linewidth='0.5', color='white')
            axs[j].set_ylabel(species_list_eng[j], rotation=0)
            axs[j].yaxis.set_label_coords(-0.45, 0)               
            for pos in not_recorded_pos: # add lines for missing data
                axs[j].axvline(pos, color='red', linestyle='-', linewidth=0.5,ymin=0, ymax=0.05)
        fig.subplots_adjust(hspace=0.3)  # Adjust the vertical space between subplots
        fig.suptitle('Vocal activity distrobution over time\nStation '+station, fontsize=10, x=0.5)
        axs[-1].set_xlabel('Time [Date]') 
         
        # Set x-axis tick positions and labels
        axs[-1].xaxis.set_major_locator(mdates.MonthLocator())  # Tick every month
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format as 'Jan 2020', 'Feb 2020', etc.
    
        axs[-1].set_xticks(range(0,len(date_index), 90))
        axs[-1].set_xticklabels(date_index[::90], rotation=90)
        axs[-1].xaxis.set_minor_locator(plt.MultipleLocator(10))   
        cbar = fig.colorbar(img, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.set_label('Relative vocal activity\n Daily calls [mean]',labelpad=-20)
        cbar.set_ticks([0,max_v])
        cbar.set_ticklabels(['low','high'])         
        red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label='Missing data') # Add custom legend
        axs[0].legend(handles=[red_line], loc=(-0.65,-33))
        #fig = plt.figure(1, figsize=(16, 16))
    return activity_distribution    
    
    
    
def activity_distribution_overview3(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
    This function calculates and plots the mean daily vocal activity for multiple species over a specified time period.
    The function reads species activity data from Excel files.
    Visualizations plot showing the vocal activity over time with imshow and highlights days with missing data.

    Parameters:
        species_list (list of str): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str or datetime.date): Start date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        date_end (str or datetime.date): End date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        tuple:
            - total_summary (pd.DataFrame): DataFrame containing the mean daily activity for each species.
    
    Usage:
        total_summary = peilut_total_summary(species_list, milon_path, species_activity_path, 
            date_start, date_end, stations)
    
    Notes:
        - The function expects an Excel file named 'time not recorded.xlsx' in the specified directory.
        - Activity Excel files must be named according to the format 'activity_<species>_<station>.xlsx'.
        - The function generates and displays plots for the vocal activity overview but does not save them.
    '''
    
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding=('iso8859_8'))   
    if type(date_start) == str:
        date_start = datetime.strptime(date_start, "%m-%d-%Y").date()
    if type(date_end) == str:
        date_end = datetime.strptime(date_end, "%m-%d-%Y").date()
    dates = pd.date_range(date_start, date_end - timedelta(days=1), freq='d')  # list of dates
    species_list_eng = []
    for s in range(len(species_list)): 
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    activity_distribution = pd.DataFrame({}, index=dates, columns=species_list_eng) 
    os.chdir(species_activity_path)
    time_not_recorded = pd.read_excel('missing_data.xlsx')
    
    for station in stations:
        for i in range(len(species_list_eng)):
            species = str(species_list_eng[i])
            file_name = 'activity_' + species + '_' + station + '.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.rename(columns={"Unnamed: 0": "date"})
            activity_distribution[species] = peilut.iloc[:, 1::].mean(axis=1).to_frame().set_axis([species_eng], axis=1).set_index(peilut['date'])

        #  Plotting:
        date_index = activity_distribution.index.date
        not_recorded = pd.to_datetime(time_not_recorded[station].dropna().tolist()).date
        not_recorded_pos = [np.where(date_index == date)[0][0] for date in not_recorded if date in date_index] 
        
        # Update the figure to make the plot wider
        fig, axs = plt.subplots(len(species_list_eng), 1, figsize=(8, 5))  
        
        for j, species in enumerate(species_list_eng):
            max_v = activity_distribution[activity_distribution.columns[len(species_list_eng)-1]].max()  # Max value of last species
            y = activity_distribution[species].values
            
            # Handling NaN values (NaN will be black)
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad(color='black')  # Set NaN values to display in black
            
            img = axs[j].imshow(np.expand_dims(y, axis=1).T, aspect='15', cmap=cmap)
            axs[j].set_yticks([])
            axs[j].set_yticklabels([])
            axs[j].set_xticks(range(0, len(date_index), 90))
            axs[j].set_xticklabels([])
            axs[j].grid(True, which='major', linestyle=':', linewidth='0.5', color='white')
            axs[j].set_ylabel(species_list_eng[j], rotation=0)
            axs[j].yaxis.set_label_coords(-0.45, 0)
            
            for pos in not_recorded_pos:  # Add lines for missing data
                axs[j].axvline(pos, color='red', linestyle='-', linewidth=1, ymin=0, ymax=0.1)
        
        fig.subplots_adjust(hspace=0.3)  # Adjust the vertical space between subplots
        fig.suptitle('Vocal activity distribution over time\nStation ' + station, fontsize=11, x=0.55)
        axs[-1].set_xlabel('Time [Date]')
        axs[-1].set_xticks(range(0, len(date_index), 90))
        axs[-1].set_xticklabels(date_index[::90], rotation=90)
        axs[-1].xaxis.set_minor_locator(plt.MultipleLocator(10))
        
        cbar = fig.colorbar(img, ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label('Relative vocal activity\n Daily calls [mean]', labelpad=-20)
        cbar.set_ticks([0, max_v])
        cbar.set_ticklabels(['low', 'high'])
        
        red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=2, label='Missing data')  # Add custom legend
        axs[0].legend(handles=[red_line], loc=(-0.57, -36))
        
    return activity_distribution    
    

def activity_distribution_overview1(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
    This function calculates and plots the mean daily vocal activity for multiple species over a specified time period.
    The function reads species activity data from Excel files.
    Visualizations plot showing the vocal activity over time with imshow and highlights days with missing data.

    Parameters:
        species_list (list of str): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str or datetime.date): Start date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        date_end (str or datetime.date): End date for the analysis in "MM-DD-YYYY" format or as a datetime.date object.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        activity_distribution (pd.DataFrame): DataFrame containing the mean daily activity for each species.
    
    Usage:
        activity_distribution = activity_distribution_overview(species_list, milon_path, species_activity_path, date_start, date_end, stations)
    '''
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding='iso8859_8')

    if isinstance(date_start, str):
        date_start = datetime.strptime(date_start, "%m-%d-%Y").date()
    if isinstance(date_end, str):
        date_end = datetime.strptime(date_end, "%m-%d-%Y").date()
    
    dates = pd.date_range(date_start, date_end - timedelta(days=1), freq='d')  # List of dates
    species_list_eng = []

    for species in species_list: 
        species_eng = milon[(milon == species).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    
    activity_distribution = pd.DataFrame({}, index=dates, columns=species_list_eng)
    os.chdir(species_activity_path)
    missing_data = pd.read_excel('missing_data.xlsx')

    fig, axs = plt.subplots(1, len(stations), figsize=(14, 8), sharey=True)

    for ax, station in zip(axs, stations):
        activity_data = pd.DataFrame(index=dates)

        for species in species_list_eng:
            file_name = f'activity_{species}_{station}.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.rename(columns={"Unnamed: 0": "date"})
            peilut['date'] = pd.to_datetime(peilut['date'])
            activity = peilut.iloc[:, 1:].mean(axis=1)
            activity.index = pd.to_datetime(peilut['date']).dt.date
            activity_data[species] = activity

        activity_data = activity_data.reindex(dates)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='black')
        img = ax.imshow(activity_data.T, aspect='auto', cmap=cmap, interpolation='nearest')
        
        not_recorded = pd.to_datetime(missing_data[station].dropna()).dt.date
        for missing_date in not_recorded:
            if missing_date in activity_data.index:
                ax.axvline(missing_date, color='red', linestyle='--', linewidth=1)

        ax.set_title(f'Station: {station}')
        ax.set_xlabel('Date')
        ax.set_xticks(range(0, len(dates), 90))
        ax.set_xticklabels(dates[::90].strftime('%Y-%m-%d'), rotation=45)
        ax.set_yticks(range(len(species_list_eng)))
        ax.set_yticklabels(species_list_eng)
        ax.grid(True, which='major', linestyle=':', linewidth='0.5', color='gray')

    fig.suptitle('Vocal Activity Distribution Over Time', fontsize=14)
    cbar = fig.colorbar(img, ax=axs, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Relative vocal activity\n Daily calls [mean]')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()

    return activity_distribution

#%% Habitat_indices (peilut_indices):
def habitat_indices(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
    peilut_indices:
        Analyze activity data for specified species and stations, and generate plots for intensity level and species abundance in habitats.
        The function reads species information, activity data, and missing time records, then processes the data to calculate and visualize indices.
        
    Parameters:
        species_list (list of str or int): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        None

    Usage:
        peilut_indices(milon_path, species_activity_path, species_list, date_start, date_end, stations)
       
    '''
    
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding=('iso8859_8'))
    #milon = milon.drop(['Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11'], axis=1)
    os.chdir(species_activity_path)
   
    # max_values = []
    sum_all_total = {}
    sum_all_bool_total = {}
    peilut_all_total = {}
    time_not_recorded = pd.read_excel('missing_data.xlsx')
    for stat in range(len(stations)):
        station = stations[stat]
        species_list_eng = []
        peilut_all = []
        peilut_all_bool = []                   
        for i in range(len(species_list)):
            species_eng = milon[(milon == species_list[i]).any(axis=1)]['species eng'].item()
            species_eng = species_eng.replace(" ", "_")
            species_list_eng.append(species_eng)            
            file_name = 'activity_'+species_eng+'_'+station+'.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.iloc[:,1:].set_index(peilut['date'])
            peilut_bool = peilut.where(peilut.isna(), np.where(peilut == 0, 0, 1))
            peilut_all.append(peilut)
            peilut_all_bool.append(peilut_bool)
        
        
            
        sum_all = pd.DataFrame(0, index=peilut_all[0].index, columns=peilut_all[0].columns)
        sum_all_bool = pd.DataFrame(0, index=peilut_all[0].index, columns=peilut_all[0].columns)
        for df in peilut_all:
            sum_all += df
        for df in peilut_all_bool:
            sum_all_bool += df
        
        sum_all_total[station] = sum_all
        sum_all_bool_total[station] = sum_all_bool  
    
    sun_times = get_sun_times(date_start, date_end)
    sun_times_dec = sun_times # sun time in decimal (for plot location)
    
    def time_to_decimal(t):
        return t.hour + t.minute / 60
    sun_times_dec['sunrise'] = pd.to_datetime(sun_times['sunrise']).dt.time.apply(time_to_decimal)
    sun_times_dec['sunset'] = pd.to_datetime(sun_times['sunset']).dt.time.apply(time_to_decimal)
    date_to_y_pos = {date: len(peilut) - 1 - idx for idx, date in enumerate(peilut.index.date)}
    
    # plotting Intensity level Index:
    for stat in range(len(stations)):
        station = stations[stat]
        #not_recorded = pd.to_datetime(time_not_recorded[station].dropna().tolist()).date
        #not_recorded_pos = [idx for idx, date in enumerate(peilut.index.date) if date in not_recorded]        
        x_pos = np.arange(len(peilut.columns))
        x_ticks_labels = peilut.columns
        y_pos = np.arange(len(peilut.index))
        y_ticks_labels = peilut.index.date        
        cmap = plt.get_cmap('viridis') 
        cmap.set_bad(color='black')
        fig, ax = plt.subplots()
        img = ax.imshow(np.flipud(sum_all_total[station]), aspect='0.05', cmap=cmap, vmin=0, vmax=int(sum_all_total[station].max().max()))
        fig.suptitle('Intensity level\nstation ' + station, x=0.67)       
        ax.set_ylabel('Time [Date]')
        ax.set_xlabel('Time [Hour]')
        
        # Select first day of each month for the x-axis ticks (only for the last subplot)
        first_of_months = pd.date_range(start=date_start, end=date_end, freq='MS')
        month_labels = [d.strftime('%b %Y') for d in first_of_months]  # Format: 'Jan 2020'
        month_positions = [np.where(peilut.index.date == d.date())[0][0] for d in first_of_months if d.date() in peilut.index.date]

          
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks_labels)
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(y_ticks_labels)
        
        ax.set_yticks(month_positions)
        ax.set_yticklabels(month_labels, rotation=0)
              
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label('Intensity level [mean number of calls]', labelpad=5)       
        # Add red lines for the dates in not_recorded[station]:
        for date in time_not_recorded[station]:
            if date.date() in peilut.index.date:
                line_position = len(peilut) - 1 - np.where(peilut.index.date == date.date())[0][0]
                ax.axhline(y=line_position, color='red', linestyle='-', linewidth=0.5, xmin=0, xmax=0.02)       
        # Get y positions for sun_times dates and add sunrise and sunset dots:
        sun_times_dec['y_pos'] = sun_times_dec['date'].map(date_to_y_pos)        
        ax.scatter(sun_times_dec['sunrise'], sun_times_dec['y_pos'], color='gray', s=0.001)
        ax.scatter(sun_times_dec['sunset'], sun_times_dec['y_pos'], color='gray', s=0.001)        
    
        red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label='Missing data') # Adding a custom legend    
        suntimes_line = plt.Line2D([], [], color='gray', linestyle='-', label='Sun set & rise',linewidth=0.5)
        ax.legend(handles=[red_line,suntimes_line], loc=(0.8,-0.35))
        
        # plotting Richness Index:
        fig, ax = plt.subplots()
        img = ax.imshow(np.flipud(sum_all_bool_total[station]), aspect='0.05', cmap=cmap, vmin=0, vmax=sum_all_bool_total[station].max().max())
        fig.suptitle('Species richness\nstation ' + station, x=0.67)       
        ax.set_ylabel('Time [Date]')
        ax.set_xlabel('Time [Hour]')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks_labels)
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.set_yticks(y_pos)
        # ax.set_yticklabels(y_ticks_labels)
        ax.set_yticks(month_positions)
        ax.set_yticklabels(month_labels, rotation=0)
        # ax.yaxis.set_major_locator(plt.MaxNLocator(20))
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
        cbar.set_label('Species richness [number of cpecies active]', labelpad=5)       
        # Add red lines for the dates in not_recorded[station]:
        for date in time_not_recorded[station]:
            if date.date() in peilut.index.date:
                line_position = len(peilut) - 1 - np.where(peilut.index.date == date.date())[0][0]
                ax.axhline(y=line_position, color='red', linestyle='-', linewidth=1, xmin=0, xmax=0.02)       
        # Get y positions for sun_times dates and add sunrise and sunset dots:
        sun_times_dec['y_pos'] = sun_times_dec['date'].map(date_to_y_pos)        
        ax.scatter(sun_times_dec['sunrise'], sun_times_dec['y_pos'], color='gray', s=0.005)
        ax.scatter(sun_times_dec['sunset'], sun_times_dec['y_pos'], color='gray', s=0.005)        
    
        red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label='Missing data') # Adding a custom legend    
        suntimes_line = plt.Line2D([], [], color='gray', linestyle='-', label='Sun set & rise',linewidth=0.5)
        ax.legend(handles=[red_line,suntimes_line], loc=(0.8,-0.35))
     
    return
               






def habitat_indices1(milon_path, species_activity_path, species_list, date_start, date_end, stations):
    '''
        Analyze activity data for specified species and stations, and generate plots for intensity level and species abundance in habitats.
        The function reads species information, activity data, and missing time records, then processes the data to calculate and visualize indices.
        
    Parameters:
        species_list (list of str or int): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str): The start date for analysis in 'YYYY-MM-DD' format.
        date_end (str): The end date for analysis in 'YYYY-MM-DD' format.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        None

    Usage:
        habitat_indices(milon_path, species_activity_path, species_list, date_start=DATE_START, date_end=DATE_END, stations=STATIONS)
    '''
    
    # os.chdir(milon_path)
    milon = pd.read_csv(os.path.join(milon_path,"milon.txt"), sep="\t", encoding=('iso8859_8'))
    
    os.chdir(species_activity_path)
   
    sum_all_total = {}
    sum_all_bool_total = {}
    missing_data = pd.read_excel('missing_data.xlsx')

    for station in stations:
        species_list_eng = []
        peilut_all = []
        peilut_all_bool = []                   
        for species in species_list:
            species_eng = milon[(milon == species).any(axis=1)]['species eng'].item()
            species_eng = species_eng.replace(" ", "_")
            species_list_eng.append(species_eng)            
            file_name = f'activity_{species_eng}_{station}.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.iloc[:, 1:].set_index(peilut['date'])
            peilut_bool = peilut.where(peilut.isna(), np.where(peilut == 0, 0, 1))
            peilut_all.append(peilut)
            peilut_all_bool.append(peilut_bool)
        
        sum_all = sum(peilut_all)
        sum_all_bool = sum(peilut_all_bool)
        sum_all_total[station] = sum_all
        sum_all_bool_total[station] = sum_all_bool  
    
    sun_times = get_sun_times(date_start, date_end,dst_transition=False)
    
    def time_to_decimal(t):
        return t.hour + t.minute / 60
    
    sun_times['sunrise'] = pd.to_datetime(sun_times['sunrise']).dt.time.apply(time_to_decimal)
    sun_times['sunset'] = pd.to_datetime(sun_times['sunset']).dt.time.apply(time_to_decimal)
    #date_to_y_pos = {date: len(peilut.index) - 1 - idx for idx, date in enumerate(peilut.index.date)}
    date_to_y_pos = {date: len(peilut.index) - 1 - idx for idx, date in enumerate(reversed(peilut.index.date))}
    
    # Create a single figure with 2x2 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 22), sharey='row')
    fig.suptitle('Habitat Indices: Intensity and Abundance', x=0.5, y=1)
    
    # Intensity Indices
    for i, station in enumerate(stations):
        ax = axs[0, i]
        x_pos = np.arange(len(sum_all_total[station].columns))
        x_ticks_labels = sum_all_total[station].columns
        y_pos = np.arange(len(sum_all_total[station].index))
        y_ticks_labels = sum_all_total[station].index.date        
        cmap = plt.get_cmap('viridis')
        img = ax.imshow(np.flipud(sum_all_total[station]), aspect='0.05', cmap=cmap)
        ax.set_title(f'Station {station}')
        ax.set_xlabel('Time [Hour]')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks_labels)
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_ticks_labels)
        ax.yaxis.set_major_locator(plt.MaxNLocator(20))
        
        # Add red lines for missing data
        for date in missing_data[station]:
            if date.date() in sum_all_total[station].index.date:
                line_position = len(sum_all_total[station]) - 1 - np.where(sum_all_total[station].index.date == date.date())[0][0]
                ax.axhline(y=line_position, color='red', linestyle='-', linewidth=1, xmin=0.48, xmax=0.52)
        
        # Add sunrise and sunset dots
        sun_times['y_pos'] = sun_times['date'].map(date_to_y_pos)        
        ax.scatter(sun_times['sunrise'], sun_times['y_pos'], color='gray', s=0.1)
        ax.scatter(sun_times['sunset'], sun_times['y_pos'], color='gray', s=0.1)        

    # Colorbar for Intensity Index
    cbar_intensity = fig.colorbar(img, ax=axs[0, :], orientation='vertical', fraction=0.03, pad=0.04)
    cbar_intensity.set_label('Intensity level [mean number of calls]', labelpad=5)

    # Abundance Indices
    for i, station in enumerate(stations):
        ax = axs[1, i]
        x_pos = np.arange(len(sum_all_bool_total[station].columns))
        x_ticks_labels = sum_all_bool_total[station].columns
        y_pos = np.arange(len(sum_all_bool_total[station].index))
        y_ticks_labels = sum_all_bool_total[station].index.date        
        cmap = plt.get_cmap('viridis')
        img = ax.imshow(np.flipud(sum_all_bool_total[station]), aspect='0.05', cmap=cmap)
        ax.set_title(f'Station {station}')
        ax.set_xlabel('Time [Hour]')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_ticks_labels)
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_ticks_labels)
        ax.yaxis.set_major_locator(plt.MaxNLocator(20))
        
        # Add red lines for missing data
        for date in missing_data[station]:
            if date.date() in sum_all_bool_total[station].index.date:
                line_position = len(sum_all_bool_total[station]) - 1 - np.where(sum_all_bool_total[station].index.date == date.date())[0][0]
                ax.axhline(y=line_position, color='red', linestyle='-', linewidth=1, xmin=0.48, xmax=0.52)
        
        # Add sunrise and sunset dots
        sun_times['y_pos'] = sun_times['date'].map(date_to_y_pos)        
        ax.scatter(sun_times['sunrise'], sun_times['y_pos'], color='gray', s=0.1)
        ax.scatter(sun_times['sunset'], sun_times['y_pos'], color='gray', s=0.1)

    # Colorbar for Abundance Index
    cbar_abundance = fig.colorbar(img, ax=axs[1, :], orientation='vertical', fraction=0.03, pad=0.04)
    cbar_abundance.set_label('Species abundance [number of species active]', labelpad=5)

    # Add common legend for missing data and sun times
    red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=3, label='Missing data')    
    suntimes_line = plt.Line2D([], [], color='gray', linestyle='-', label='Sun set & rise', linewidth=0.5)
    fig.legend(handles=[red_line, suntimes_line], loc='upper right')

    #fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    #fig.tight_layout(pad=0, w_pad=-0.5, h_pad=1.0)
    #fig.tight_layout(rect=[0, 0, 1, 0.95], pad=2, w_pad=0.5, h_pad=1)
    #fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0, hspace=0.3)
    #fig.subplots_adjust(pad=0, w_pad=0.1, h_pad=1.0)
    #fig.tight_layout()

    plt.show()

#%% Species_level_overview (peilut_total_species):
def species_level_overview(milon_path,species_activity_path,species_list,date_start,date_end,stations):
    '''
    peilut_total_species:
    This function reads species activity data from Excel files, calculates the mean daily activity and standard error of the 
    mean (SEM), and creates visualizations showing the vocal activity over time. It also highlights days with missing data.

    Parameters:
        species_list (list of str): List of species names or IDs for which activity data should be analyzed.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        date_start (str or datetime.date): Start date for the analysis in "M-D-Y" format or as a datetime.date object.
        date_end (str or datetime.date): End date for the analysis in "M-D-Y" format or as a datetime.date object.
        stations (list of str): List of station names to include in the analysis.

    Returns:
        dict: A dictionary where each key is a station name and each value is another dictionary with the following keys:
            - 'mean' (pd.DataFrame): DataFrame containing the mean daily activity for each species.
            - 'ste' (pd.DataFrame): DataFrame containing the standard error of the mean (SEM) for daily activity.
    
    Usage:
        total_summary = species_level_overview(milon_path,species_activity_path,species_list=SPECIES_LIST,date_start=DATE_START,date_end=DATE_END,stations=STATIONS)

    
    Notes:
        - The function expects an Excel file named 'time not recorded.xlsx' in the specified directory.
        - Activity Excel files must be named according to the format 'activity_<species>_<station>.xlsx'.
        - The function calculate and return the SEM but wont display it in the plot.
        - The function generates and displays plots for the vocal activity overview but does not save them.
    '''
    
    # os.chdir(milon_path)
    milon = pd.read_csv(os.path.join(milon_path,"milon.txt"), sep="\t", encoding=('iso8859_8'))
    if type(date_start) == str:
        date_start = datetime.strptime(date_start, "%m-%d-%Y").date()
    if type(date_end) == str:
        date_end = datetime.strptime(date_end, "%m-%d-%Y").date()
    dates = pd.date_range(date_start,date_end-timedelta(days=1),freq='d') #list of dates
    species_list_eng = []
    for s in range(len(species_list)): 
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    t_mean = pd.DataFrame({}, index=dates, columns=species_list_eng) 
    t_ste = pd.DataFrame({}, index=dates, columns=species_list_eng)
    total_summary = {station: {'mean': t_mean.copy(), 'ste': t_ste.copy()} for station in stations}
    os.chdir(species_activity_path)
    if not os.path.isfile('missing_data.xlsx'):
        raise FileNotFoundError("The file 'time not recorded.xlsx' is not found, to create it use peilut.peilut_indices with not_recorded=True")   
    missing_data = pd.read_excel('missing_data.xlsx')

    # plot:
    for i in range(len(species_list_eng)):
        species = str(species_list_eng[i])      
        fig, axs = plt.subplots(2)
        for stat in range(len(stations)):
           station = stations[stat]
           date_index = total_summary[station]['mean'].index.date
           not_recorded = pd.to_datetime(missing_data[station].dropna().tolist()).date
           not_recorded_pos = [np.where(date_index == date)[0][0] for date in not_recorded if date in date_index] 
           file_name = 'activity_'+species+'_'+station+'.xlsx'
           peilut = pd.read_excel(file_name)
           peilut = peilut.rename(columns={"Unnamed: 0": "date"})
           total_summary[station]['mean'][species] = peilut.iloc[:, 1::].mean(axis=1).to_frame().set_axis([species_eng], axis=1).set_index(peilut['date'])
           total_summary[station]['ste'][species] = peilut.iloc[:, 1::].sem(axis=1).to_frame().set_axis([species_eng], axis=1).set_index(peilut['date'])   
           x_pos = np.arange(total_summary[station]['mean'].shape[0])          
           y = total_summary[station]['mean'][species].values
           date_index = total_summary[station]['mean'].index.date                     
           axs[stat].bar(x_pos,y,width=1)
           fig.suptitle(f'Activity over time\n{species}', fontsize=12, y=0.99)
           axs[stat].set_title(f'station {station}')
           # axs[stat].set_xticks(range(0,len(date_index), 90))
           axs[stat].set_xticklabels([])
           # axs[stat].xaxis.set_minor_locator(plt.MultipleLocator(10))
           axs[stat].yaxis.set_minor_locator(plt.MultipleLocator(2))
           axs[stat].grid(True, which='major', linestyle=':', linewidth='0.5', color='gray')
           # axs[stat].set_title('station '+station)     
           for pos in not_recorded_pos: # Add vertical lines for missing data
               axs[stat].axvline(x=pos, color='red', linestyle='-', linewidth=1, ymin=0., ymax=0.02)                  
        axs[-1].set_ylabel('Calls per hour [Mean]')
        axs[-1].yaxis.set_label_coords(-0.08, 1)
        
        axs[-1].set_xlabel('Time [Date]') 
        first_of_months = pd.date_range(start=date_start, end=date_end, freq='MS')
        month_labels = [d.strftime('%b %Y') for d in first_of_months]  # Format: 'Jan 2020'
        month_positions = [np.where(date_index == d.date())[0][0] for d in first_of_months if d.date() in date_index]
            
        axs[-1].set_xlabel('Time [Date]')
        axs[-1].set_xticks(month_positions)
        axs[-1].set_xticklabels(month_labels, rotation=90)
        # axs[-1].set_xticks(range(0,len(date_index), 90))
        # axs[-1].set_xticklabels(date_index[::90], rotation=90)
        custom_lines = [plt.Line2D([0], [0], color='red', linestyle='-', linewidth=2)] # Create a custom legend entry for missing data
        fig.legend(custom_lines, ['Missing data'], loc='lower right', bbox_to_anchor=(0.2, 1))
        
    return total_summary

#%% Activity_image (peilut img):
def activity_image(milon_path,species_activity_path,species_list,stations):
    '''
    Generate activity images for specified species across multiple stations.
    This function reads species information and activity data,
    creates activity images for each species at each station,
    and plots them with additional annotations for missing data and sunrise/sunset times.

    Parameters:
        species_list (list of str or int): List of species names or IDs to visualize.
        milon_path (str): Path to the directory containing the 'milon.txt' file with species information.
        species_activity_path (str): Path to the directory containing activity Excel files and 'time not recorded.xlsx'.
        stations (list of str): List of station names to include in the visualization.

    Returns:
        peilut_all (dict): Nested dictionary containing activity images for each species at each station.

    Usage:
        peilut_all = activity_image(milon_path,species_activity_path,species_list=SPECIES_LIST,stations=STATIONS)

    '''
    
    # os.chdir(milon_path)
    milon = pd.read_csv(os.path.join(milon_path,"milon.txt"), sep="\t", encoding=('iso8859_8'))
    #milon = milon.drop(['Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11'], axis=1)
    os.chdir(species_activity_path)
    missing_data = pd.read_excel('missing_data.xlsx')    
    species_list_eng = []
    for s in range(len(species_list)): 
        species_eng = milon[(milon == species_list[s]).any(axis=1)]['species eng'].item()
        species_eng = species_eng.replace(" ", "_")
        species_list_eng.append(species_eng)
    
    peilut_all = dict.fromkeys(stations)
    for stat in range(len(stations)):
        station = stations[stat]
        peilut_all[station] = {}
        for i in range(len(species_list_eng)):
            file_name = 'activity_'+species_list_eng[i]+'_'+station+'.xlsx'
            peilut = pd.read_excel(file_name)
            peilut = peilut.iloc[:, 1:].set_index(peilut["date"])
            peilut_all[station][species_list_eng[i]] = peilut
    
    date_start = peilut.index[0].strftime('%m-%d-%Y')
    date_end = peilut.index[-1].strftime('%m-%d-%Y')
    sun_times = get_sun_times(date_start, date_end)
    sun_times_dec = sun_times # sun time in decimal (for plot location)
    
    def time_to_decimal(t):
        return t.hour + t.minute / 60
    sun_times_dec['sunrise'] = pd.to_datetime(sun_times['sunrise']).dt.time.apply(time_to_decimal)
    sun_times_dec['sunset'] = pd.to_datetime(sun_times['sunset']).dt.time.apply(time_to_decimal)
    sun_times_dec['y_pos'] = sun_times_dec.index
    #date_to_y_pos = {date: len(peilut) - 1 - idx for idx, date in enumerate(peilut.index.date)}
   
    # plotting:
    x_pos = np.arange(len(peilut.columns))
    x_ticks_labels = peilut.columns
    y_pos = np.arange(len(peilut.index))
    y_ticks_labels = peilut.index.date
    cmap = plt.get_cmap('viridis') 
    for stat in range(len(stations)):
        station = stations[stat] 
        for j in range(len(species_list_eng)):
            fig, ax = plt.subplots()
            img = ax.imshow(np.flipud(peilut_all[station][species_list_eng[j]]), aspect='0.05', cmap=cmap)
            # img = ax.imshow(peilut_all[station][species_list_eng[j]], aspect='0.05', cmap=cmap)
            fig.suptitle(f'Activity image\n{species_list_eng[j]} - station{station}', x=0.65, y=1)       
            ax.set_ylabel('Time [Date]')
            ax.set_xlabel('Time [Hour]')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_ticks_labels)
            ax.xaxis.set_major_locator(plt.MaxNLocator(12))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(y_ticks_labels)
            ax.yaxis.set_major_locator(plt.MaxNLocator(20))
            cbar = fig.colorbar(img, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
            cbar.set_label('Activity level [Hourly mean number of calls]', labelpad=5)       
            # Add red lines for the dates in not_recorded[station]:
            for date in missing_data[station]:
                if date.date() in peilut.index.date:
                    line_position = len(peilut) - 1 - np.where(peilut.index.date == date.date())[0][0]
                    ax.axhline(y=line_position, color='red', linestyle='-', linewidth=1, xmin=0.48, xmax=0.52)       
            # Get y positions for sun_times dates and add sunrise and sunset dots:
            #sun_times_dec['y_pos'] = sun_times_dec['date'].map(date_to_y_pos)        
            ax.scatter(sun_times_dec['sunrise'], sun_times_dec['y_pos'], color='gray', s=0.005)
            ax.scatter(sun_times_dec['sunset'], sun_times_dec['y_pos'], color='gray', s=0.005)        
        
            red_line = plt.Line2D([], [], color='red', linestyle='-', linewidth=3, label='Missing data') # Adding a custom legend    
            suntimes_line = plt.Line2D([], [], color='gray', linestyle='-', label='Sun set & rise',linewidth=0.5)
            ax.legend(handles=[red_line,suntimes_line], loc=(1.1,-0.32))
            
    return peilut_all

#%% peilut hours change - daily_activity_changes:
def peilut_hours_change(input_species,milon_path,species_activity_path,date_start,date_end,window_size,window_hop,stations,interval=1):
    """
    peilut_hours_change
     
    This function calculates and displays the daily activity changes of selected species over a specified period. 
    It generates plots to visualize the mean and standard error of the daily activity for each station over the 
    given time windows.
        
    Parameters:
        input_species (str or int): Name or ID number of the species.
        milon_path (str): Path to the milon text file.
        species_activity_path (str): Path to the directory containing activity Excel files for all species.
        date_start (str or datetime): Start date in 'MM-DD-YYYY' format or as a datetime object.
        date_end (str or datetime): End date in 'MM-DD-YYYY' format or as a datetime object.
        window_size (int): Size of the time window in days.
        window_hop (int): Hop size in days for the sliding window.
        stations (list of str): List of station names.
        interval (int, optional): Interval in hours for aggregating data. Default is 1 hour.

     Usage:
        peilut_hours_change(input_species, milon_path, species_activity_path, 
            date_start, date_end, window_size, window_hop, 
            stations, interval)       
    """
    os.chdir(milon_path)
    milon = pd.read_csv("milon.txt", sep="\t", encoding=('iso8859_8'))
    milon = milon.drop(['Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11'], axis=1)
    species_eng = milon[(milon == input_species).any(axis=1)]['species eng'].item()
    species_eng = species_eng.replace(" ", "_")
    os.chdir(species_activity_path)
    if type(date_start) == str:
        initial_date_start = datetime.strptime(date_start, '%m-%d-%Y')
    elif type(date_start) == datetime:
        initial_date_start = date_start
    if type(date_end) == str:
        date_end = datetime.strptime(date_end, '%m-%d-%Y')        
    for stat in range(len(stations)):
        station = stations[stat]
        file_name = 'activity_'+species_eng+'_'+station+'.xlsx'
        peilut = pd.read_excel(file_name)
        peilut = peilut.rename(columns={"Unnamed: 0": "date"})
        peilut_mean = pd.DataFrame()
        peilut_ste = pd.DataFrame()        
        row_index_start = peilut['date'].searchsorted(initial_date_start)
        row_index_end = peilut['date'].searchsorted(initial_date_start + relativedelta(days=window_size))
        date_start = initial_date_start        
        while row_index_end < len(peilut):
            the_mean = peilut.loc[row_index_start:row_index_end].iloc[:,1:].mean(axis=0).to_frame(name=date_start).T
            the_ste = peilut.loc[row_index_start:row_index_end].iloc[:,1:].sem(axis=0).to_frame(name=date_start).T
            peilut_mean = pd.concat([peilut_mean, the_mean])
            peilut_ste = pd.concat([peilut_ste, the_ste])   
            date_start = date_start + relativedelta(days=window_hop) # changing date_start by adding hop
            row_index_start = peilut['date'].searchsorted(date_start) # updating row index
            row_index_end = peilut['date'].searchsorted(date_start + relativedelta(days=window_size)) # updating row index
        # Aggregate the data by the chosen interval:
        num_intervals = peilut_mean.shape[1] // interval
        peilut_mean_agg = peilut_mean.groupby(np.arange(peilut_mean.columns.size) // interval, axis=1).mean()
        peilut_ste_agg = peilut_ste.groupby(np.arange(peilut_ste.columns.size) // interval, axis=1).mean()              
        x_pos = range(num_intervals)
        x_ticks_labels = [peilut_mean.columns[i*interval] for i in range(num_intervals)]
        for j in range(peilut_mean_agg.shape[0]):
            window_dates = peilut_mean.index[j].strftime('%Y-%m-%d')+"  -  "+(peilut_mean.index[j] + relativedelta(days=window_size)).strftime('%Y-%m-%d')
            y = peilut_mean_agg.iloc[j]
            y_err = peilut_ste_agg.iloc[j]
            fig, ax = plt.subplots()
            ax.bar(x_pos,y,yerr=y_err,capsize=3)
            ax.set_title('Daily vocal activity', pad=30)
            plt.suptitle(species_eng+' - station '+station+'\n'+window_dates+" ("+str(window_size)+" days)", fontsize=10, y=0.97)
            ax.set_ylabel('Vocal activity level\nNumber of calls [Mean]')
            ax.set_xlabel('Time ['+str(interval)+' Hours interval]')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_ticks_labels)
            ax.set_ylim(0, peilut_mean_agg.max().max()*1.05)
            ax.grid(True, which='major', linestyle=':', linewidth='0.5', color='gray')
            #ax.grid(True, which='minor', linestyle=':', linewidth='0.5', color='gray')   
    return 