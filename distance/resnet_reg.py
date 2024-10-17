# -*- coding: utf-8 -*-
 '''
    AUTOFOCUSING REGRESSION MODEL FOR DHM
    
    Residual Neural Network (ResNet) for a regression task (distances from -10 to 10).
    All distances are normalized to the range -1,1 before training.

    Python Version: 3.10.12
    Keras version: 3.3.2
    Tensorflow vesion: 2.16.1

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 17/10/2024
'''

# Import libraries
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

####################################################################################################################################
#%% MODEL DETAILS
# ARCHITECTURE TYPE: Sequential Semantic Segmentation Regression Network 
arch_type = "Resnet model 3 regression + transfer learning. 2048 neurons"
model_filename = "model_reg_resnet8.h5"
loss_image = "model_reg_resnet8.png"

transfer_model = 'model_class_resnet8.h5' # resnet model 98% acc

# Image specifications
img_dimensions = (512, 512)

# Defining hyperparameters (epochs)
epoch = 400
batch = 64
lr = 0.00001  # value in ITM paper 0.00008


# Define callbacks (EarlyStopping and checkpoint)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(filepath="checkpoint/model_amp_{epoch:02d}.keras",
                                      save_best_only=True,
                                      monitor='val_loss',
                                      verbose=1,
                                      mode='min')

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
            
            processed_images.append((normalized_img, distance))
        except Exception as e:
            print(f"Error loading or normalizing image {filename}: {e}")

    return processed_images


def split_data(processed_images):
    images, distances = zip(*processed_images)
    images = np.array(images)
    
    # Convert distances to numeric type
    distances = np.array(distances, dtype=int)

    # Normalize distances
    real_dist, norm_dist, min_distance, max_distance = normalize_distances(distances)
    print(f"Minimum distance:{min_distance}")
    print(f"Maximum distance:{max_distance}")
    
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(images, norm_dist, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), real_dist, norm_dist

####################################################################################################################################
#%% SOME MODEL DEFINITIONS (CALLBACKS, FOURIER TRANSFORM LAYER)

# Custom callback to save weights after each epoch and track time per epoch
class SaveWeightsAndTimeCallback(Callback):
    def __init__(self, initial_epoch=0, lr=0.00001, batch_size=64, arch_type="Resnet"):
        super(SaveWeightsAndTimeCallback, self).__init__()
        self.initial_epoch = initial_epoch
        self.lr = lr
        self.batch_size = batch_size
        self.arch_type = arch_type

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - self.epoch_start_time
        hours, remainder = divmod(epoch_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        actual_epoch = epoch + self.initial_epoch + 1  # Adjust for initial_epoch
        print(f'Epoch {actual_epoch} took {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')

        # Save weights with a detailed filename
        filename = f'weights/weights_epoch_{actual_epoch}.weights.h5'
        self.model.save_weights(filename)
        print(f'Weights saved for epoch {actual_epoch} as {filename}')

        
# Create directory for weights if not exists
if not os.path.exists('weights'):
    os.makedirs('weights')

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

####################################################################################################################################
#%% TRANSFER LEARNING ARCHITECTURE

def class_to_reg(model_path, FourierTransformLayer, lr):
    
    # Load the pre-trained model
    class_model = tf.keras.models.load_model(model_path, custom_objects={'FourierTransformLayer': FourierTransformLayer}, compile=False)
    
    # Freeze all layers up to and including the 512-neuron dense layer
    for layer in class_model.layers:
        layer.trainable = True
    
    # Access the output of the 2048-neuron dense layer
    x = class_model.get_layer('dense_2048').output

    # Add a Dense layer with linear activation for regression
    x = Dense(2048, activation='relu', name='dense_2048_2')(x)
    
    x = Dense(512, activation='relu',name='dense_512')(x)

    # Add dropout after the 512-neuron dense layer
    x = Dropout(0.5, name='last_dropout')(x)
    
    # Apply tanh activation to the output
    new_output = Dense(1, activation='tanh', name='tanh_activation')(x)
    #new_output = Activation('tanh', name='tanh_activation')(x)
    
    # Create new model
    regression_model = Model(inputs=class_model.input, outputs=new_output)
    
    # Compile the new model with a regression-specific loss and optimizer
    regression_model.compile(optimizer=Adam(learning_rate=lr), 
                             loss='mean_squared_error',  # Loss function for regression
                             metrics=['mean_absolute_error'])
    
    return regression_model

####################################################################################################################################
#%% PREPARING DATA

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

####################################################################################################################################
#%% TRANSFER LEARNING IMPLEMENTATION

regression_model = class_to_reg(transfer_model, FourierTransformLayer, lr)

# To verify the model summary
regression_model.summary()
    
#%% TRAINING THE MODEL

# Compile the model with a regression loss function
regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                         loss='mean_squared_error', 
                         metrics=['mae'])

# Load weights from the last epoch if available
if os.path.exists('weights'):
    latest_weights = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir('weights') if f.endswith('.weights.h5')], default=0)
    if latest_weights > 0:
        regression_model.load_weights(f'weights/weights_epoch_{latest_weights}.weights.h5')
        print(f"Loaded weights from epoch {latest_weights}")
    else:
        print("No previous weights found, starting from scratch")
else:
    print("Weights directory not found, starting from scratch")


# Initialize the callback with model details
time_tracking_callback = SaveWeightsAndTimeCallback(
    initial_epoch=latest_weights,
    lr=lr,
    batch_size=batch,
    arch_type=arch_type
)
    

# Training the model
start_time = datetime.now()

history = regression_model.fit(X_train, y_train, epochs=epoch, batch_size=batch,
                               validation_data=(X_val, y_val), 
                               callbacks=[early_stopping, time_tracking_callback])

end_time = datetime.now()

# Calculate the training duration
duration = end_time - start_time
hours, remainder = divmod(duration.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

# Save the new regression model

regression_model.save(model_filename)

# Evaluate the model
test_loss, test_mae = regression_model.evaluate(X_test, y_test)

####################################################################################################################################
#%% MODEL RESULTS

def plot_loss_and_metric(lr, history, filename=loss_image):
    """
    Plot both the training and validation loss, as well as accuracy, on the same plot, and save it to a file.

    Args:
        lr (float): Learning rate used in training.
        history (History): Training history object.
        filename (str): Filename to save the combined plot.
    """
    plt.figure(figsize=(12, 6))

    # Plotting Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Plotting Accuracy
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')

    plt.title(f'Model Loss and Accuracy Progression. lr={lr}')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Combined loss and accuracy plot saved as {filename}")

# Plotting the training and validation loss
plot_loss_and_metric(lr, history)

print("-------------------------------------------------------------------")
print("-**** MODEL TRAINING SUMMARY ****-")
print(f'Architecture type: {arch_type}')
print(f'Learning rate: {lr}')
print(f'Batch size: {batch}')
print(f'(On test dataset) Loss: {test_loss}, MSE: {test_mae}')

print(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
print(f'Model saved as {model_filename}')
print(f"Plot saved as {loss_image}")
print("-------------------------------------------------------------------")
