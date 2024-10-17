'''
    AUTOFOCUSING RES-UNET MODEL FOR DHM AMPLITUDE RECONSTRUCTIONS
    
    This model is designed for amplitude image reconstruction from holographic images using a U-Net architecture enhanced with ResNet blocks. 
    The model follows an encoder-decoder structure and incorporates several convolutional layers, skip connections, and residual blocks.

    Inputs: Out-of-focus simulated holograms
    Outputs: In-focus amplitude reconstructions.
    
    Python Version: 3.10.12
    Keras version: 3.3.2
    Tensorflow vesion: 2.16.1

    Author: Maria Paula Rey*, Raul Castañeda**
    Applied Sciences and Engineering School, EAFIT University (Applied Optics Group)  
    Email: *mpreyb@eafit.edu.co , **racastaneq@eafit.edu.co
    
    Date last modified: 17/10/2024
'''

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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2DTranspose, UpSampling2D, Add, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Add, BatchNormalization, Activation, Concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.preprocessing import normalize
from tensorflow.keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator

####################################################################################################################################
#%% MODEL DETAILS

# ARCHITECTURE TYPE: Sequential Semantic Segmentation Regression Network 
arch_type = "U-net Resnet model with data augmentation. Amp reconstruction"
model_filename = "model_unet_amp_2.h5"
loss_image = "model_unet_amp_2.png"


# Image specifications
img_dimensions = (512, 512)

# Defining hyperparameters (epochs)
epoch = 200
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

# Batch data loading class
class HoloAmpDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, holo_files, amp_files, batch_size=16, target_size=(512, 512), shuffle=False, augment=True):
        self.holo_files = holo_files
        self.amp_files = amp_files
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augment = augment
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
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, holo_batch, amp_batch):
        X = np.empty((self.batch_size, *self.target_size, 1))
        y = np.empty((self.batch_size, *self.target_size, 1))
        for i, (holo_file, amp_file) in enumerate(zip(holo_batch, amp_batch)):
            holo_img = tf.keras.preprocessing.image.load_img(holo_file, color_mode='grayscale', target_size=self.target_size)
            holo_img = tf.keras.preprocessing.image.img_to_array(holo_img) / 255.0

            amp_img = tf.keras.preprocessing.image.load_img(amp_file, color_mode='grayscale', target_size=self.target_size)
            amp_img = tf.keras.preprocessing.image.img_to_array(amp_img) / 255.0

            # Apply data augmentation
            if self.augment:
                holo_img, amp_img = self.apply_augmentation(holo_img, amp_img)

            X[i,] = holo_img
            y[i,] = amp_img
        
        return X, y
    
    def apply_augmentation(self, image1, image2):
    # Random flip left-right
        if random.random() > 0.5:
            image1 = np.fliplr(image1)
            image2 = np.fliplr(image2)

        # Random flip up-down
        if random.random() > 0.5:
            image1 = np.flipud(image1)
            image2 = np.flipud(image2)

        # Random rotation of 90, 180, or 270 degrees
        rotation_choice = np.random.choice([90, 180, 270])
        k = rotation_choice // 90  # Number of times to rotate by 90 degrees
        image1 = np.rot90(image1, k=k)
        image2 = np.rot90(image2, k=k)

        return image1, image2
    
#%%

def split_data(holo_files, amp_files, train_percent=0.8, val_percent=0.1):
    
    print("Splitting data... ")
    # Calculate test percentage
    test_percent = 1 - train_percent - val_percent
    
    # Split into training+validation and test
    train_val_holo, test_holo, train_val_amp, test_amp = train_test_split(
        holo_files, amp_files, test_size=test_percent, random_state=42
    )
    
    # Split training+validation into training and validation
    train_holo, val_holo, train_amp, val_amp = train_test_split(
        train_val_holo, train_val_amp, test_size=val_percent/(train_percent + val_percent), random_state=42
    )
    
    return (train_holo, train_amp), (val_holo, val_amp), (test_holo, test_amp)


####################################################################################################################################
#%% SOME MODEL DEFINITIONS 

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

####################################################################################################################################   
#%% RESIDUAL BLOCK DEFINITION

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

####################################################################################################################################
#%% MAIN ARCHITECTURE

def resnet_unet(lr, input_shape=(512, 512, 1)):
    inputs = Input(shape=input_shape)

    # Encoder-----------------------------------------------------------------------
    # Initial convolution and pooling
    x = Conv2D(64, (7, 7), padding='same', strides=2, activation='relu')(inputs)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Store skip connections in a list
    skips = []

    # Residual blocks with downsampling
    # Block 1
    for _ in range(3):
        x = ResBlock(x, filters=64)
    skips.append(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    for _ in range(4):
        x = ResBlock(x, filters=128)
    skips.append(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3
    for _ in range(6):
        x = ResBlock(x, filters=256)
    skips.append(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 4
    for _ in range(3):
        x = ResBlock(x, filters=512)
    skips.append(x)
    x = MaxPooling2D((2, 2))(x)

    # Bottleneck
    for _ in range(3):
        x = ResBlock(x, filters=1024)

    # Decoder-----------------------------------------------------------------------
    # Reverse the order of skip connections
    skips = skips[::-1]

    # Block 1
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skips[0]])
    for _ in range(3):
        x = ResBlock(x, filters=512)

    # Block 2
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skips[1]])
    for _ in range(6):
        x = ResBlock(x, filters=256)

    # Block 3
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skips[2]])
    for _ in range(4):
        x = ResBlock(x, filters=128)

    # Block 4
    x = UpSampling2D((2, 2))(x)
    x = Concatenate()([x, skips[3]])
    for _ in range(3):
        x = ResBlock(x, filters=64)

    # Final upsampling to match the input size
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)

    # Output layer
    outputs = Conv2D(input_shape[-1], (1, 1), activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
                  loss='mse',  # Mean squared error for reconstruction
                  metrics=['mae'])

    return model
  
####################################################################################################################################
#%%

# Function to make sure holo and amp are aligned
def extract_key(fname):
    match = re.match(r'(\d+)_(\d+)_(\w)_(-?\d+)\.png', os.path.basename(fname))
    if match:
        digit = int(match.group(1))
        number = int(match.group(2))
        distance = int(match.group(4))
        return (digit, number, distance)
    return None

#%% PREPARING DATA
# Paths to hologram and amplitude images
holo_dir = 'HOLO_V2.0'  
amp_dir = 'AMP_V2.0'

batch_size = 16
target_size = (512, 512)

# Load file names
holo_files = [os.path.join(holo_dir, fname) for fname in os.listdir(holo_dir) if fname.endswith('.png')]
amp_files = [os.path.join(amp_dir, fname) for fname in os.listdir(amp_dir) if fname.endswith('.png')]

# Sort the lists based on extracted key
holo_files.sort(key=extract_key)
amp_files.sort(key=extract_key)

# Ensure lengths match
assert len(holo_files) == len(amp_files), "Mismatch between number of hologram and amplitude files"

# Perform data splitting
(train_holo_files, train_amp_files), (val_holo_files, val_amp_files), (test_holo_files, test_amp_files) = split_data(holo_files, amp_files)

# Create data generators
train_gen = HoloAmpDataGenerator(train_holo_files, train_amp_files, batch_size=batch_size, target_size=target_size)
val_gen = HoloAmpDataGenerator(val_holo_files, val_amp_files, batch_size=batch_size, target_size=target_size)
test_gen = HoloAmpDataGenerator(test_holo_files, test_amp_files, batch_size=batch_size, target_size=target_size)


# Print data loading summary
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

print("----------")

# Fetch a batch from the generator
X_batch, y_batch = train_gen[0]  # Fetch the first batch

# Print shapes of the batches
print("Shape of hologram batch (X_batch):", X_batch.shape)
print("Shape of amplitude batch (y_batch):", y_batch.shape)

print("-------------------------------------------------------------------")

####################################################################################################################################
#%% MAKING SURE HOLO AND AMP ARE ALIGNED (NOT NEEDED FOR EVERY TIME YOU RUN)

def save_random_three_from_generator(data_generator, save_dir, num_samples=3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Fetch one batch from the generator
    X_batch, y_batch = data_generator[0]  # Fetch the first batch
    
    # Randomly select samples
    sampled_indices = np.random.randint(1, 16, size=num_samples)

    for i in sampled_indices:
        holo_img = X_batch[i]
        amp_img = y_batch[i]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Hologram')
        plt.imshow(np.squeeze(holo_img), cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Amplitude')
        plt.imshow(np.squeeze(amp_img), cmap='gray')
        plt.axis('off')

        save_path = os.path.join(save_dir, f'sample_{i}.png')
        plt.savefig(save_path)
        plt.close()

save_random_three_from_generator(train_gen, 'MATCH')

####################################################################################################################################
#%% MODEL INITIALIZATION

input_shape = (512, 512, 1)  # Example input shape, adjust as needed

amp_model = resnet_unet(lr)
amp_model.summary()


#%% TRAINING MODEL
# Load weights from the last epoch if available
if os.path.exists('weights'):
    latest_weights = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir('weights') if f.endswith('.weights.h5')], default=0)
    if latest_weights > 0:
        amp_model.load_weights(f'weights/weights_epoch_{latest_weights}.weights.h5')
        print(f"Loaded weights from epoch {latest_weights}")
    else:
        print("No previous weights found, starting from scratch")
else:
    print("Weights directory not found, starting from scratch")
    
# Instantiate the callback
time_tracking_callback = SaveWeightsAndTimeCallback()
    
# Training the model
start_time = datetime.now()

# model training
history = amp_model.fit(train_gen, initial_epoch=latest_weights, epochs=epoch, batch_size=batch,
                    validation_data=val_gen, verbose=1,
                    callbacks=[early_stopping, time_tracking_callback])

end_time = datetime.now()

# Calculate the training duration
duration = end_time - start_time
hours, remainder = divmod(duration.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)

# Save the model
amp_model.save(model_filename)


####################################################################################################################################
# MODEL RESULTS
def plot_loss_and_metric(lr, history, filename=loss_image):
    """
    Plot both the training and validation loss, as well as MAE, on the same plot, and save it to a file.

    Args:
        lr (float): Learning rate used in training.
        history (History): Training history object.
        filename (str): Filename to save the combined plot.
    """
    plt.figure(figsize=(12, 6))

    # Plotting Loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Plotting MAE
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')

    plt.title(f'Model Loss and MAE Progression (lr={lr})')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Combined loss and MAE plot saved as {filename}")


# Plotting the training and validation loss
plot_loss_and_metric(lr, history)

print("-------------------------------------------------------------------")
print("-**** MODEL TRAINING SUMMARY ****-")
print(f'Architecture type: {arch_type}')
print(f'Learning rate: {lr}')
print(f'Batch size: {batch}')
print(f"Training took {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
print(f'Model saved as {model_filename}')
print(f"Plot saved as {loss_image}")
print("-------------------------------------------------------------------")
