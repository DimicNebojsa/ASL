import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
input_path = '/Users/nebojsadimic/Downloads/ASL_dataset/asl_alphabet_train/asl_alphabet_train'

# Image dimensions
IMG_SIZE = (224, 224)  
BATCH_SIZE = 32

# # Data Augmentation using ImageDataGenerator
# datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,  # Normalize images
#     rotation_range=20,   # Random rotation
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     validation_split=0.2  # Splitting dataset into training and validation
# )

# # Train data generator
# train_generator = datagen.flow_from_directory(
#     input_path,
#     target_size=IMG_SIZE,  # Updated to 224x224
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training'
# )

# # Validation data generator
# val_generator = datagen.flow_from_directory(
#     input_path,
#     target_size=IMG_SIZE,  # Updated to 224x224
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation'
# )

# # Get number of classes
# num_classes = len(train_generator.class_indices)
# print(f"Number of classes: {num_classes}")


data_dir = '/Users/nebojsadimic/Downloads/ASL_dataset/asl_alphabet_train/asl_alphabet_train'
target_size = (64, 64)
target_dims = (64, 64, 3) # add channel for RGB
n_classes = 29
val_frac = 0.1
batch_size = 64

data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    validation_split=val_frac)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")
val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")

my_model = Sequential()
my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=target_dims))
my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
my_model.add(Dropout(0.5))
my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))
my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))
my_model.add(Flatten())
my_model.add(Dropout(0.5))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(n_classes, activation='softmax'))

my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

my_model.fit(train_generator, epochs=5, validation_data=val_generator)


# # CNN Model Architecture (Same as before)
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Updated Input Shape
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile Model
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train Model
# model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=10
# )

# # Evaluate Model
# test_loss, test_acc = model.evaluate(val_generator)
# print(f"Test Accuracy: {test_acc:.4f}")
