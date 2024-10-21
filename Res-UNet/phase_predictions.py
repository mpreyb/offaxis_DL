# -*- coding: utf-8 -*-
"""phase_predictions.py

# PREDICTIONS FOR AMPLITUDE RECONSTRUCTIONS
--------------------------------------------------------------------------------
Regression model trained using transfer learning from a classification model.
Includes autoencoder for amplitude reconstruction.

Author: Maria Paula Rey*, Raul Castañeda**

Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
Email: mpreyb@eafit.edu.co , racastaneq@eafit.edu.co

Date last modified: 30/09/2024
"""
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Layer
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import mean_squared_error
import time

#%% MODEL DETAILS

model_path = "model_unet_pha.h5"
output_dir = "pha_pred_unet_3"

img_dimensions = (512, 512)

epoch = 100
batch = 260
lr =  0.00001

#%% DATA LOADING FUNCTIONS

class HoloAmpDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, holo_files, amp_files, batch_size=16, target_size=(512, 512), shuffle=False):
        self.holo_files = holo_files
        self.amp_files = amp_files
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.holo_files) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        holo_batch = [self.holo_files[i] for i in indexes]
        amp_batch = [self.amp_files[i] for i in indexes]
        X, y = self.__data_generation(holo_batch, amp_batch)
        return X, y
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.holo_files))
        self.indexes = np.arange(len(self.amp_files))
    
    def __data_generation(self, holo_batch, amp_batch):
        X = np.empty((self.batch_size, *self.target_size, 1))
        y = np.empty((self.batch_size, *self.target_size, 1))
        for i, (holo_file, amp_file) in enumerate(zip(holo_batch, amp_batch)):
            holo_img = tf.keras.preprocessing.image.load_img(holo_file, color_mode='grayscale', target_size=self.target_size)
            holo_img = tf.keras.preprocessing.image.img_to_array(holo_img) / 255.0
            X[i,] = holo_img
            
            amp_img = tf.keras.preprocessing.image.load_img(amp_file, color_mode='grayscale', target_size=self.target_size)
            amp_img = tf.keras.preprocessing.image.img_to_array(amp_img) / 255.0
            
            y[i,] = amp_img
        
        return X, y

#%%

def split_data(holo_files, amp_files, train_percent=0.8, val_percent=0.1):
    print("Splitting data... ")
    test_percent = 1 - train_percent - val_percent
    train_val_holo, test_holo, train_val_amp, test_amp = train_test_split(
        holo_files, amp_files, test_size=test_percent, random_state=42
    )
    train_holo, val_holo, train_amp, val_amp = train_test_split(
        train_val_holo, train_val_amp, test_size=val_percent/(train_percent + val_percent), random_state=42
    )
    
    return (train_holo, train_amp), (val_holo, val_amp), (test_holo, test_amp)

#%% FOURIER TRANSFORM LAYER
    
class FourierTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(FourierTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        fft = tf.signal.fft2d(tf.cast(inputs, tf.complex64))
        magnitude = tf.abs(fft)
        return magnitude

    def compute_output_shape(self, input_shape):
        return input_shape
    
#%% RESNET BLOCK
def ResBlock(inputs, filters, kernel_size=3, stride=1, padding='same', mode='encode'):
    """
    Residual block function.
    
    Parameters:
    - inputs: Input tensor
    - filters: Number of filters for Conv2D or Conv2DTranspose
    - kernel_size: Size of the convolution kernel
    - stride: Stride for convolution
    - padding: Padding for convolution
    - mode: 'encode' for Conv2D or 'decode' for Conv2DTranspose
    
    Returns:
    - Output tensor after passing through the residual block
    """
    
    assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."

    if mode == 'encode':
        conv_layer = Conv2D
    elif mode == 'decode':
        conv_layer = Conv2DTranspose
    
    # First convolutional layer
    x = conv_layer(filters, kernel_size, strides=stride, padding=padding)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolutional layer
    x = conv_layer(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    if inputs.shape[-1] != filters or stride != 1:
        shortcut = conv_layer(filters, (1, 1), strides=stride, padding=padding)(inputs)
    else:
        shortcut = inputs
    
    # Add the shortcut to the main path
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

#%%

def extract_key(fname):
    match = re.match(r'(\d+)_(\d+)_(\w)_(-?\d+)\.png', os.path.basename(fname))
    if match:
        digit = int(match.group(1))
        number = int(match.group(2))
        distance = int(match.group(4))
        return (digit, number, distance)
    return None

#%% PREPARING DATA

holo_dir = 'HOLO_V2.0'  
pha_dir = 'PHA_V2.0'  

batch_size = 32
target_size = (512, 512)

holo_files = [os.path.join(holo_dir, fname) for fname in os.listdir(holo_dir) if fname.endswith('.png')]
pha_files = [os.path.join(pha_dir, fname) for fname in os.listdir(pha_dir) if fname.endswith('.png')]

holo_files.sort(key=extract_key)
pha_files.sort(key=extract_key)

assert len(holo_files) == len(pha_files), "Mismatch between number of hologram and amplitude files"

(train_holo_files, train_amp_files), (val_holo_files, val_amp_files), (test_holo_files, test_amp_files) = split_data(holo_files, pha_files)

train_gen = HoloAmpDataGenerator(train_holo_files, train_amp_files, batch_size=batch_size, target_size=target_size)
val_gen = HoloAmpDataGenerator(val_holo_files, val_amp_files, batch_size=batch_size, target_size=target_size)
test_gen = HoloAmpDataGenerator(test_holo_files, test_amp_files, batch_size=batch_size, target_size=target_size)

print("-------------------------------------------------------------------")
print("-**** DATA LOADING SUMMARY ****-")
print(f"Size of images: {target_size}")
print(" Data splitting:")
print("- Number of holograms for training", len(train_holo_files))
print("- Number of holograms for validation", len(val_holo_files))
print("- Number of holograms for test", len(test_holo_files))
print("- Number of amplitudes for training", len(train_amp_files))
print("- Number of amplitudes for validation", len(val_amp_files))
print("- Number of amplitudes for test", len(test_amp_files))
print("-----------")

X_batch, y_batch = train_gen[0]

print("Shape of hologram batch (X_batch):", X_batch.shape)
print("-------------------------------------------------------------------")

#%% SSIM

def compute_ssim_per_distance(test_gen, predictions):
    distance_ssim_map = {}
    ssim_scores_per_distance = {}

    for i in range(len(test_gen)):
        test_samples, test_amp = test_gen[i]
        for j in range(len(test_samples)):
            distance = extract_key(test_gen.holo_files[i * test_gen.batch_size + j])[2]
            true_image = test_amp[j, ..., 0]
            predicted_image = predictions[i * test_gen.batch_size + j, ..., 0]
            score, _ = ssim(true_image, predicted_image, full=True, data_range=1.0)
            
            if distance not in ssim_scores_per_distance:
                ssim_scores_per_distance[distance] = []
            ssim_scores_per_distance[distance].append(score)
    
    for distance, scores in ssim_scores_per_distance.items():
        distance_ssim_map[distance] = np.mean(scores)
    
    return distance_ssim_map, ssim_scores_per_distance

def plot_ssim_distribution(ssim_scores_per_distance, output_path="ssim_distribution.png"):
    distances = sorted(ssim_scores_per_distance.keys())
    ssim_values = [ssim_scores_per_distance[dist] for dist in distances]

    plt.figure(figsize=(10, 6))
    plt.boxplot(ssim_values, labels=distances, showmeans=True)
    plt.title('SSIM Distribution per Distance')
    plt.xlabel('Distance')
    plt.ylabel('SSIM')
    plt.grid(True)
    
    # Save the plot as a .png file
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory
    
#%% LOADING MODEL AND MAKING PREDICTIONS

unet = load_model(model_path, custom_objects={'ResBlock': ResBlock}, compile=False)
unet.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=['mae'])

#unet.compile(optimizer=Adam(learning_rate=0.00001 ), loss='mean_squared_error')

start_time = time.time()
predictions = unet.predict(test_gen)
end_time = time.time()

# Evaluate the model on the training set
test_mse, test_mae = unet.evaluate(test_gen, verbose=1)
print(f"Mean Squared Error (MSE) on training data: {test_mse}")
print(f"Mean Absolute Error (MAE) on training data: {test_mae}")

elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Predictions completed in: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

#%% Compute SSIM and Plot Results

distance_ssim_map, ssim_scores_per_distance = compute_ssim_per_distance(test_gen, predictions)

# Print average SSIM for all predictions
all_ssim_scores = [score for scores in ssim_scores_per_distance.values() for score in scores]
average_ssim = np.mean(all_ssim_scores)
print(f"Average SSIM for all predictions: {average_ssim}")

# Save SSIM Distribution plot as a PNG file
plot_ssim_distribution(ssim_scores_per_distance, output_path=os.path.join(output_dir, "ssim_distribution_phases.png"))

#%%

def compute_ssim(true_images, predicted_images):
    """
    Computes the Structural Similarity Index (SSIM) between two sets of images.

    Parameters:
    - true_images: numpy array of ground truth images
    - predicted_images: numpy array of predicted images

    Returns:
    - ssim_scores: list of SSIM scores for each pair of images
    """
    ssim_scores = []

    for i in range(len(true_images)):
        # Assuming the images are in floating-point format and normalized between 0 and 1
        score, _ = ssim(true_images[i, ..., 0], predicted_images[i, ..., 0], full=True, data_range=1.0)
        ssim_scores.append(score)

    return ssim_scores

# Compute SSIM for all the predictions in the first batch
X_test, y_test = test_gen[0]  # Fetch the first batch
ssim_scores = compute_ssim(y_test, predictions)
#average_ssim = np.mean(ssim_scores)

# Extract a batch of data from the test generator
test_samples, test_pha = test_gen.__getitem__(0)  # Get the first batch

# Making predictions
predicted_pha = unet.predict(test_samples)

# Save the results for each sample in the batch
for i in range(len(test_samples)):
    plt.figure(figsize=(12, 6))

    # Extract distance information
    distance = extract_key(test_gen.holo_files[i])[2]

    # Display original hologram
    plt.subplot(1, 3, 1)
    plt.imshow(test_samples[i].reshape(512, 512), cmap='gray')
    plt.title(f'Distance: {distance} mm\nOriginal Hologram {i+1}')

    # Display true amplitude
    plt.subplot(1, 3, 2)
    plt.imshow(test_pha[i].reshape(512, 512), cmap='gray')
    plt.title(f'True Amplitude {i+1}')

    # Display predicted amplitude
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_pha[i].reshape(512, 512), cmap='gray')
    plt.title(f'Predicted Phase {i+1} (SSIM: {ssim_scores[i]:.3f})')

    # Save the plot as a .png file
    plt.savefig(os.path.join(output_dir, f'prediction_{i+1}.png'))
    plt.close()  # Close the figure to free memory
