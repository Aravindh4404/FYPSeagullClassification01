# -*- coding: utf-8 -*-
"""Initial_try_keras.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-T54feuzrQ1eFtcclS1s51u6yC5qObMw
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/content/drive')


import os
from shutil import copy2
import csv
import tensorflow as tf



from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # The %tensorflow_version magic only works in colab.
#   %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt


!pip install tensorflow_hub


import tensorflow_hub as hub


tf.__version__


import pandas as pd

# Increase precision of presented data for better side-by-side comparison
pd.set_option("display.precision", 8)

# Set the data root directory
data_root = "/content/drive/My Drive/FYP/Dataset/train"

# Image size for resizing
IMAGE_SHAPE = (224, 224)

# Create ImageDataGenerators with validation split
datagen_kwargs = dict(rescale=1./255, validation_split=.20)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

# Define training and validation generators
train_generator = train_datagen.flow_from_directory(
    data_root,  # Directory for training data
    subset="training",
    shuffle=True,
    target_size=IMAGE_SHAPE,
    class_mode='categorical',
    batch_size=32
)

valid_generator = valid_datagen.flow_from_directory(
    data_root,  # Directory for validation data
    subset="validation",
    shuffle=True,
    target_size=IMAGE_SHAPE,
    class_mode='categorical',
    batch_size=32
)

# Print the class indices to verify
print("Class Indices (Training):", train_generator.class_indices)
print("Class Indices (Validation):", valid_generator.class_indices)

# Save labels to a text file
labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
    f.write(labels)

# Print the labels
!cat labels.txt

import tensorflow as tf

# Define the input shape
input_shape = (224, 224, 3)

# Load the MobileNetV2 model pre-trained on ImageNet, excluding the top layers
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',  # Load pre-trained weights
    include_top=False,   # Exclude the top (fully-connected) layers
    input_shape=input_shape
)

# Freeze the base model layers
base_model.trainable = False

# Define the model using the Functional API
inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Display model summary
model.summary()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define the number of epochs
epochs = 40

# Train the model
history = model.fit(
    train_generator,
    validation_data=valid_generator,  # Use valid_generator if you have validation data
    epochs=epochs,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size if valid_generator else None,
    verbose=1
)

"""Save trained model check performance"""

test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
print(f"Validation Loss: {test_loss}")
print(f"Validation Accuracy: {test_acc}")

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Save the model

model.save('my_model.h5')
model.save('/content/drive/My Drive/FYP/MODELS/trained_model.h5')

from google.colab import files
files.download('my_model.h5')

# Load the model (when needed)
loaded_model = tf.keras.models.load_model('my_model.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img_path = '/content/1200 - 2024-08-18T191835.473.jpeg'
img = image.load_img(img_path, target_size=input_shape[:2])
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Scale the image

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Print the predicted class
print(f"Predicted class: {list(train_generator.class_indices.keys())[predicted_class[0]]}")