# -*- coding: utf-8 -*-
"""

#  CLASSIFICATION PREDICTIONS (Resnet)
--------------------------------------------------------------------------------
Distance classification (21 categories) 

Author: Maria Paula Rey*, Raul Castañeda**

Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
Email: mpreyb@eafit.edu.co , racastaneq@eafit.edu.co

Date last modified: 21/10/2024
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import re
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

#%%

#filename = 'try.png'
img_dimensions = (512, 512)

#model_filename = "model_class_resnet_7.h5"
model_filename = "model_class_resnet7_2048.h5"


#%%

class FourierTransformLayer(Layer):
    def __init__(self, **kwargs):
        """
        Initialize the custom FourierTransformLayer.
        """
        super(FourierTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Apply 2D Fast Fourier Transform (FFT) to the input images.
        
        Args:
            inputs (Tensor): Input tensor of shape (batch_size, height, width, channels).
        
        Returns:
            Tensor: Magnitude of the 2D FFT, which serves as the transformed feature map.
        """
        # Perform 2D FFT on the input tensor
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        # Compute the magnitude of the FFT, which is a measure of frequency content
        magnitude = tf.abs(fft)
        return magnitude

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer.
        
        Args:
            input_shape (tuple): Shape of the input tensor.
        
        Returns:
            tuple: Shape of the output tensor, which is the same as input_shape.
        """
        return input_shape

#%%

# Function to create the categories
def map_distance_to_category(distance):
    distance = int(distance)
    category = distance + 10  # Shift the range [-10, 10] to [0, 20]
    
    if 0 <= category <= 20:
        return category
    else:
        raise ValueError(f"Distance {distance} out of expected range.")


# Function for loading images and obtaining the corresponding propagation distance
def load_data(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            images.append(filename)
    return images


# Function to create the instances with filenames
def create_instances(image_names):
    instances = []
    filenames = []
    for image in image_names:
        # Using regex to split the name for the pattern with single "-"
        match_single = re.match(r'(\d+)_(\d+)_(\w+)_(-?\d+)\.png', image)
        if match_single:
            third_digit = match_single.group(4)
            category = map_distance_to_category(third_digit)
            instances.append((image, category))
            filenames.append(image)
    return instances, filenames

# Function to visualize images and their respective instances
def display_images(instances, dir_images, num_images):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        index = random.randint(0, len(instances) - 1)
        image_name, category = instances[index]
        image_path = os.path.join(dir_images, image_name)

        try:
            img = Image.open(image_path).convert('L')
            axes[i].imshow(np.asarray(img), cmap='gray')
            axes[i].set_title(f'{image_name}\nCategory: {category}')
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")

    plt.show()
    return


# Function for processing (normalize) images
def processing_data(instances, dir_images, image_size):
    processed_images = []
    for filename, category in instances:
        image_path = os.path.join(dir_images, filename)
        try:
            img = Image.open(image_path).convert('L')
            img = img.resize(image_size)
            img_array = np.asarray(img)
            
            normalized_img = img_array / 255.0
            processed_images.append((normalized_img, category))
        except Exception as e:
            print(f"Error loading or normalizing image {filename}: {e}")

    return processed_images


# Function to split data and retain filenames
def split_data(processed_images, all_filenames):
    images, categories = zip(*processed_images)
    images = np.array(images)
    categories = np.array(categories)
    
    X_train, X_temp, y_train, y_temp, train_filenames, temp_filenames = train_test_split(
        images, categories, all_filenames, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test, val_filenames, test_filenames = train_test_split(
        X_temp, y_temp, temp_filenames, test_size=0.5, random_state=42
    )
    
    return (X_train, y_train, train_filenames), (X_val, y_val, val_filenames), (X_test, y_test, test_filenames)

#%%


# Function to generate predictions and save to CSV
def save_predictions_to_csv(model, X_test, y_test, test_filenames, csv_filename):
    """
    Generate predictions for test data and save to a CSV file.

    Args:
        model (tf.keras.Model): The trained model.
        X_test (np.ndarray): The test images.
        y_test (np.ndarray): The true labels for the test images.
        test_filenames (list): The filenames corresponding to the test images.
        csv_filename (str): The filename for the CSV file to save the results.
    """
    # Predict the categories for the test images
    predictions = model.predict(X_test)
    predicted_categories = np.argmax(predictions, axis=1)
    
    # Create a DataFrame with filenames, true labels, and predicted labels
    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': predicted_categories,
        'Filename': test_filenames
    })
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_categories)
    print(f"Accuracy: {accuracy:.4f}")

#%%

# path to load data
dir_images = 'HOLO_V2.0'

# Loading images data and pre-processing
holo_data = load_data(dir_images)

# Creating instances and filenames
instances, all_filenames = create_instances(holo_data)

# Normalize images
processed_images = processing_data(instances, dir_images, img_dimensions)

# Split data and retain filenames
(holo_train, cat_train, train_filenames), (holo_val, cat_val, val_filenames), (holo_test, cat_test, test_filenames) = split_data(processed_images, all_filenames)

# Load the trained model
model = tf.keras.models.load_model(model_filename, custom_objects={'FourierTransformLayer': FourierTransformLayer})

# Correctly evaluate the model
test_loss, test_accuracy = model.evaluate(holo_test, cat_test, verbose=1)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Ensure test_filenames corresponds to the actual test images
#test_filenames = [filename for filename, _ in instances if filename in holo_test_filenames]

# Save predictions and calculate accuracy
save_predictions_to_csv(model, holo_test, cat_test, test_filenames, 'test_predictions_2048.csv')


print("---------------------------------------------------")
print(f"Predictions done using {model_filename}")

print("---------------------------------------------------")


