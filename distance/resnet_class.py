'''
    CLASSIFICATION MODEL
    
    Residual Neural Network (ResNet) for a classification task with 21 categories.
    The primary objective of this model is to facilitate autofocusing solutions in Digital Holographic Microscopy (DHM). 
    To identify the in-focus plane, we perform transfer learning using this model onto the regression model (see resnet_reg.py)
    
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
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Add, BatchNormalization, Activation
import tensorflow as tf
from tensorflow.keras.layers import Layer

#%%   MODEL DETAILS

# ARCHITECTURE TYPE: Residual Network
arch_type = "Resnet model, 21 categories."
model_filename = "model_class_resnet8.h5"
loss_image = "model_class_resnet8.png"

# Image specifications
img_dimensions = (512, 512)

# Defining hyperparameters (epochs)
epoch = 200
batch = 64
lr = 0.00001 #value in ITM paper 0.00008

# Define callbacks (EarlyStopping and checkpoint)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint(filepath="checkpoint/model_amp_{epoch:02d}.keras",
                                      save_best_only=True,
                                      monitor='val_loss',
                                      verbose=1,
                                      mode='min')

####################################################################################################################################
# NECESSARY FUNCTIONS

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


# Function to create the instances
def create_instances(image_names):
    instances = []
    for image in image_names:
        # Using regex to split the name for the pattern with single "-"
        match_single = re.match(r'(\d+)_(\d+)_(\w+)_(-?\d+)\.png', image)
        if match_single:
            third_digit = match_single.group(4)
            category = map_distance_to_category(third_digit)
            instances.append((image, category))
    return instances


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


# Function for split images and its categories
def split_data(processed_images):
    images, categories = zip(*processed_images)
    images = np.array(images)
    categories = np.array(categories)

    X_train, X_temp, y_train, y_temp = train_test_split(images, categories, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

####################################################################################################################################

# Custom callback to save weights after each epoch and track time per epoch
class SaveWeightsAndTimeCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - self.epoch_start_time
        hours, remainder = divmod(epoch_duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f'Epoch {epoch + 1} took {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')
        
        self.model.save_weights(f'weights/weights_epoch_{epoch + 1}.weights.h5')
        print(f'Weights saved for epoch {epoch + 1}')
        
# Create directory for weights if not exists
if not os.path.exists('weights'):
    os.makedirs('weights')

####################################################################################################################################

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

def resnet_block(input_tensor, filters, kernel_size=3):
    """
    Define a ResNet block consisting of two convolutional layers with residual connections.
    
    Args:
        input_tensor (Tensor): Input tensor to the ResNet block.
        filters (int): Number of filters for the convolutional layers.
        kernel_size (int): Size of the convolutional kernel. Default is 3.
    
    Returns:
        Tensor: Output tensor after applying the ResNet block operations.
    """
    # First convolutional layer followed by batch normalization and ReLU activation
    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer followed by batch normalization
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # If the number of filters in the input tensor differs from the output, apply a 1x1 convolution to match dimensions
    if input_tensor.shape[-1] != filters:
        input_tensor = Conv2D(filters, (1, 1), padding='same')(input_tensor)

    # Add the input tensor to the output of the block (residual connection) and apply ReLU activation
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

####################################################################################################################################
def resnet_model_ft(lr, input_shape=(512, 512, 1)):
    
    inputs = Input(shape=input_shape)
    
    x = FourierTransformLayer()(inputs)     #Fourier transform of hologram
    x = Conv2D(64, (7, 7), padding='same', strides=2, activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(3):
        x = resnet_block(x, 64)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(4):
        x = resnet_block(x, 128)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(6):
        x = resnet_block(x, 256)
    x = MaxPooling2D((2, 2))(x)

    for _ in range(3):
        x = resnet_block(x, 512) 
        
    x = MaxPooling2D((2, 2))(x)  

    x = Flatten()(x)
    x = Dense(2048, activation='relu', name='dense_2048')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(21, name='21_neurons')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

####################################################################################################################################
#%% PREPARING DATA

# path to load data
dir_images = 'HOLO_V2.0'

# loading images data and pre-processing
holo_data = load_data(dir_images)

# creating instances
instances = create_instances(holo_data)
# print(instances)

# display images and its instances
#display_images(instances, dir_images, num_images=5)

# normalize images
processed_images = processing_data(instances, dir_images, img_dimensions)
# print(processed_images)

# split data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(processed_images)

# Reshape the data to include the channel dimension
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print("-------------------------------------------------------------------")
print("-**** DATA LOADING SUMMARY ****-")
print(f"Size of images: {img_dimensions}")

print(" Data splitting:")
print("- Number of holograms for training", len(X_train))
print("- Number of holograms for validation", len(X_val))
print("- Number of holograms for test", len(X_test))
print("-------------------------------------------------------------------")

####################################################################################################################################
#%% MODEL TRAINING

model = resnet_model_ft(lr)
model.summary()

# Load weights from the last epoch if available
if os.path.exists('weights'):
    latest_weights = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir('weights') if f.endswith('.weights.h5')], default=0)
    if latest_weights > 0:
        model.load_weights(f'weights/weights_epoch_{latest_weights}.weights.h5')
        print(f"Loaded weights from epoch {latest_weights}")
    else:
        print("No previous weights found, starting from scratch")
else:
    print("Weights directory not found, starting from scratch")
    
# Instantiate the callback
time_tracking_callback = SaveWeightsAndTimeCallback()
    
# Training the model
start_time = datetime.now()

history = model.fit(X_train, y_train, initial_epoch=latest_weights, epochs=epoch, batch_size=batch,
                    validation_data=(X_val, y_val), verbose=1,
                    callbacks=[early_stopping, time_tracking_callback])

end_time = datetime.now()

# Calculate the training duration
duration = end_time - start_time
hours, remainder = divmod(duration.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

# Save the model
model.save(model_filename)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

####################################################################################################################################
#%% MODEL RESULTS

def plot_loss_and_accuracy(lr, history, filename=loss_image):
    """
    Plot both the training and validation loss, as well as accuracy, on the same plot but with different y-axes,
    and save it to a file.

    Args:
        lr (float): Learning rate used in training.
        history (History): Training history object.
        filename (str): Filename to save the combined plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting Loss on the left y-axis
    ax1.plot(history.history['loss'], label='Train Loss', color='tab:blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Creating a second y-axis for Accuracy
    ax2 = ax1.twinx()
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='tab:green')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='tab:red')
    ax2.set_ylabel('Accuracy', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    # Title and legend
    plt.title(f'Model Loss and Accuracy Progression. lr={lr}')
    fig.tight_layout()

    # Saving the plot
    plt.savefig(filename)
    plt.close()
    print(f"Combined loss and accuracy plot with different y-axes saved as {filename}")

plot_loss_and_accuracy(lr, history)

print("-------------------------------------------------------------------")
print("-**** MODEL TRAINING SUMMARY ****-")
print(f'Architecture type: {arch_type}')
print(f'Learning rate: {lr}')
print(f'Batch size: {batch}')
print(f'(On test dataset) Loss: {loss}, ACCURACY: {accuracy}')

print(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
print(f'Model saved as {model_filename}')
print(f"Plot saved as {loss_image}")
print("-------------------------------------------------------------------")
