AgmoNET

This project aims to classify bird calls and analyze species activity using AI and machine learning techniques. The system uses various functions to preprocess audio data, train models, and analyze bird activity across different stations.


Project Structure:

AgmoNET/
│
├── main.py # Main script to run the project
├── requirements.txt # List of required Python packages
├── README.md # Project documentation
├── resources/
│ ├── species_mapping.txt # Mapping file for species IDs and names (milon)
│ └── species_mapping.pkl # Mapping file for species IDs and names (milon)
│
├── data/
│ ├── input/ # Folder for input raw audio data
│ ├── output/ # Folder for processed data and results
│ └── models/ # Folder for saving trained models
│
├── src/
│ ├── init.py # Initialization file for the src package
│ ├── data_processing.py # Functions for processing data reading, segmenting, augmenting, and converting to spectrograms
│ ├── modeling.py # Functions for building, training, and evaluating models
│ ├── activity_analysis.py # Functions for analyzing species activity
│ └── utilities.py # Utility functions, including saving species mapping to DataFrame
│
└── notebooks/ # Jupyter notebooks for exploration and analysis


Getting Started:

Prerequisites:
Python 3.8 or higher
Required Python packages listed in requirements.txt

Installation:
Clone the repository:
git clone https://github.com/yourusername/bird_recognition_project.git
cd bird_recognition_project

Install the required packages:
pip install -r requirements.txt

File Setup:
Place your input data in the data/input/ directory.
Pre-trained models and processed data will be saved in the data/models/ and data/output/ directories, respectively.
The resources/species_mapping.txt file contains species ID mappings and should be used for species-related functions.

Usage:
Running the Main Script:
Execute main.py to run the full pipeline: from data preprocessing to model training and evaluation.
python main.py

Analyzing Species Activity:
The activity_analysis.py module contains functions to analyze species activity across different stations. Example functions include:

peilut_compar: Compare activity levels of specified species.
peilut_total_summary: Calculate and plot mean daily vocal activity over time.
Model Training and Evaluation:
Use modeling.py to build and train models. The main function build_compile utilizes the simple_resnet architecture to compile a model for classification tasks.

Key Functions:
Data Processing (data_processing.py):
read_files: Reads audio data from specified paths.
segment_data: Segments the audio data into specified durations.
augment_data: Applies augmentations to the training data for increased variety.

Modeling (modeling.py):
build_compile: Builds and compiles a convolutional neural network model.
plot_history: Plots the training history of the model.
model_evaluate: Evaluates the trained model on test data.

Activity Analysis (activity_analysis.py):
peilut_compar: Compares species activity across stations.
peilut_total_summary: Summarizes species activity over time.

Saving Species Mapping
A script in species_mapping.py reads the species_mapping.txt file, processes the species information, and saves it as a DataFrame for easy access across functions.

Contributing:
Feel free to submit issues or pull requests to help improve this project.

License:
This project is free to use

Acknowledgments:
Thanks to all contributors and the broader AI community for providing tools and resources.
Special thanks to...