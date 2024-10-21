# -*- coding: utf-8 -*-
"""
# PREDICTIONS ON TEST DATASET (REGRESSION MODEL)

Author: Maria Paula Rey*, Raul Castañeda**

Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
Email: mpreyb@eafit.edu.co , racastaneq@eafit.edu.co

Date last modified: 21/10/2024
"""

# Import libraries
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import os
from sklearn.model_selection import train_test_split
import time
import re
import tensorflow as tf
from tensorflow.keras.layers import Layer


# Trained model path
model_name = "model_resnet_reg.h5"   

# .csv file to save predictions
pred_file = 'pred_model_resnet.csv'    

img_dimensions = (512, 512)

####################################################################################################################################
# NECESSARY FUNCTIONS

class FourierTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(FourierTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Perform 2D FFT
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        # Get the magnitude (absolute value) of the FFT
        magnitude = tf.abs(fft)
        return magnitude

    def compute_output_shape(self, input_shape):
        return input_shape


####################################################################################################################################
#%% DATA LOADING FUNCTIONS
 
# Function for loading images and obtaining the corresponding propagation distance
def load_data(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            images.append(filename)
    return images


# Function to create the instances
def create_instances(image_names):
    instances = []
    for image in image_names:
        # Using regex to split the name for the pattern with single "-"
        match_single = re.match(r'(\d+)_(\d+)_(\w+)_(-?\d+)\.png', image)
        if match_single:
            third_digit = match_single.group(4)
            #category = map_distance_to_category(third_digit)
            distance = third_digit
            #print(f"Extracted distance: {distance}")
            instances.append((image, distance))
    return instances


# Normalize distances across the entire dataset before splitting
def normalize_distances(distances):
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    distances_norm = (distances - min_distance) / (max_distance - min_distance)  # Normalize to range 0 to 1
    distances_norm = (distances_norm * 2) - 1  # Scale to range -1 to 1
    
    print(f"Real dist: {distances}, Norm dist: {distances_norm}")
    
    return distances, distances_norm, min_distance, max_distance

# Function for processing (normalize) images
def processing_data(instances, dir_images, image_size):
    processed_images = []
    for filename, distance in instances:
        image_path = os.path.join(dir_images, filename)
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize(image_size)
            img_array = np.asarray(img)
            
            normalized_img = img_array / 255.0
            
            # Add channel dimension (grayscale images will have 1 channel)
            #normalized_img = np.expand_dims(normalized_img, axis=-1)
            
            processed_images.append((normalized_img, distance))
        except Exception as e:
            print(f"Error loading or normalizing image {filename}: {e}")

    return processed_images


def split_data(processed_images):
    images, distances = zip(*processed_images)
    images = np.array(images)
    
    # Convert distances to numeric type
    distances = np.array(distances, dtype=int)
    
    real_dist, norm_dist, min_distance, max_distance = normalize_distances(distances)
    print(f"Minimum distance:{min_distance}")
    print(f"Maximum distance:{max_distance}")
    
    # Normalize distances using the entire dataset
    #normalized_distances, min_distance, max_distance = normalize_distances(distances)
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(images, norm_dist, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), real_dist, norm_dist

####################################################################################################################################

# path to load data
dir_images = 'HOLO_V2.0'

# loading images data and pre-processing
holo_data = load_data(dir_images)

# creating instances
instances = create_instances(holo_data)

# normalize images
processed_images = processing_data(instances, dir_images, img_dimensions)

# split data and get the normalization factors
(X_train, y_train), (X_val, y_val), (X_test, y_test), real_dist, norm_dist = split_data(processed_images)

# Reshape the data to include the channel dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("-------------------------------------------------------------------")
print("-**** DATA LOADING SUMMARY ****-")
print(f"Size of images: {img_dimensions}")

print(" Data splitting:")
print("- Number of amplitudes for training", len(X_train))
print("- Number of amplitudes for validation", len(X_val))
print("- Number of amplitudes for test", len(X_test))
print("-------------------------------------------------------------------")

#%%
####################################################################################################################################
#%% LOADING MODEL AND MAKING PREDICTIONS

# Load the model with the custom layer
model = load_model(model_name, custom_objects={'FourierTransformLayer': FourierTransformLayer})

# Make predictions with the model
start_time = time.time()

# Predict on the test dataset
predicted_distances = model.predict(X_test)
end_time = time.time()

predicted_distances = predicted_distances.flatten()

# Calculate absolute errors
rel_errors = np.abs(y_test - predicted_distances) / np.abs(y_test)

# Create a DataFrame to save to CSV
results_df = pd.DataFrame({
    'Target Distance': y_test,
    'Predicted Distance': predicted_distances,
    'Rel Error': rel_errors
})

# Save the DataFrame to a CSV file
results_df.to_csv(pred_file, index=False)

# Timing and performance details
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Predictions completed in: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

####################################################################################################################################
#%% REPORT

print("-------------------------------------------------------------------")
print("-**** MODEL PREDICTIONS SUMMARY ****-")

print(f"Results saved to {pred_file}.")
print(f"Predictions completed in: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

print("-------------------------------------------------------------------")
