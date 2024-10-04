# -*- coding: utf-8 -*-
"""


@author: 
"""

#%% IMPORT PACKAGES:
#import glob, os
import numpy as np
import pandas as pd
# pd.options.mode.copy_on_write = True
import seaborn as sns
#import tensorflow as tf
from tensorflow.keras import layers, models, Input
from plot_keras_history import plot_history
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
#import datetime
#import time
#import librosa, librosa.display
#import matplotlib.pyplot as plt

#%% Simple ResNet Model:
def simple_resnet(input_shape, num_classes):
    """
    Constructs a simple version of a ResNet model.
    
    Parameters:
        input_shape (tuple): The shape of the input images, typically (height, width, channels).
        num_classes (int): The number of output classes for the classification task.

    Returns:
        model (keras.Model): A Keras model instance with the defined architecture.

    Usage:
        model = simple_resnet(input_shape, num_classes)       
    """
    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x       
        # First layer of the residual block
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)        
        # Second layer of the residual block
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)       
        # Add the shortcut to the output of the second conv layer
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)      
        return x    
    inputs = Input(shape=input_shape)   
    # Initial Conv Layer
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(x)    
    # Residual Blocks
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)    
    # Global Average Pooling and Dense Layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)    
    # Build the model
    model = models.Model(inputs, outputs)
    return model

#%% model build & compile:
def build_compile(dataset,model_function=simple_resnet):
    """
    Prepares the data and builds a Convolutional Neural Network (CNN) model using the `simple_resnet` architecture. 
    The function compiles the model with the Adam optimizer and sparse categorical cross-entropy loss function.

    Parameters:
        dataset (DataFrame): A pandas DataFrame containing the signals (2D arrays in 'signal column') and their corresponding labels (ints in 'y' column).
        model_function (function): A function that creates and returns a CNN model when called.

    Returns:
        model (keras.Model): The compiled CNN model.
        X (ndarray): A 4D numpy array containing the signal data reshaped for the CNN model.
        y (ndarray): A 1D numpy array of class labels corresponding to the signals.

    Usage:
    model, X, y = build_compile(dataset)
    """
    X = np.moveaxis(np.stack(dataset['signal'], axis=1), 0, 1) # stacking list of 2D arrays to 3D array and changing axis
    X = X[...,np.newaxis] # add new axis for CNN models
    y = dataset['y'].values      
    input_shape = (X[0].shape[0],X[0].shape[1],1) # use for CNN models only: (features,timeleps,kernels)
    num_classes = len(dataset['y'].unique())       
    model = model_function(input_shape=input_shape, num_classes=num_classes) # Create the model
    model.summary()
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model, X, y




#%% plot history:
def plot_history(model_history):
    """
    Plots training and validation loss and accuracy curves from model history.

    Parameters:
        model_history (keras.callbacks.History): The history object contains the training and validation loss and accuracy.

    Usage:
        plot_history(model_history)
    """    
    history_df = pd.DataFrame(model_history.history) # Convert history to a DataFrame   
    sns.set_theme(style="whitegrid") # Set the style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))    
    # Plot Loss:
    sns.lineplot(data=history_df[['loss', 'val_loss']], ax=ax1)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')    
    # Plot Accuracy:
    sns.lineplot(data=history_df[['accuracy', 'val_accuracy']], ax=ax2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')    
    plt.tight_layout() # Adjust layout
    plt.show()


# # Evaluate the model:
# loss, accuracy = model.evaluate(x_train, y_train)
# print(f'Loss: {loss}, Accuracy: {accuracy}')

#%% model evaluate:
def model_evaluate(model, X_test, y_test, y_to_species):
    """    
    Evaluate a trained model on the test dataset and visualize the results.

    Parameters:
    - model: The trained classification model. This should be a Keras model or any other classification model that has a `predict` method.
    - X_test: The test features, provided as a numpy array or pandas DataFrame. These features are used to make predictions with the model.
    - y_test: The true labels for the test data, provided as a numpy array or pandas Series. These are the ground truth labels for evaluation purposes.
    - y_to_species: A dictionary that maps class indices to species names. This is used to label the axes of the confusion matrix.

    Returns:
    - dict: A dictionary containing the following evaluation metrics:
        - 'accuracy': The accuracy score of the model on the test data.
        - 'precision': The precision score, averaged across all classes.
        - 'recall': The recall score, averaged across all classes.
        - 'f1_score': The F1-score, averaged across all classes.
        - 'confusion_matrix': The confusion matrix showing the performance of the model.
    
    Usage:
       evaluation_results = model_evaluate(model, X_test, y_test, y_to_species) 
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # For models that predict probabilities, choose the class with the highest probability
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = y_pred.argmax(axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Display classification report
    print(classification_report(y_test, y_pred))
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
           
    # Plot the confusion matrix
    class_names = [y_to_species[i] for i in range(len(y_to_species))]  # Extract species names in order
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return results