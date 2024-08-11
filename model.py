import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Dataset paths
path1 = 'D:/vs code/datasets/facial recognition/dataset/Happy'
path2 = 'D:/vs code/datasets/facial recognition/dataset/Angry'
path3 = 'D:/vs code/datasets/facial recognition/dataset/Neutral'
path4 = 'D:/vs code/datasets/facial recognition/dataset/Sad'
path5 = 'D:/vs code/datasets/facial recognition/dataset/Surprise'

# Collecting file paths
path1_files = [os.path.join(path1, filename) for filename in os.listdir(path1)[:1000]]
path2_files = [os.path.join(path2, filename) for filename in os.listdir(path2)[:1000]]
path3_files = [os.path.join(path3, filename) for filename in os.listdir(path3)[:1000]]
path4_files = [os.path.join(path4, filename) for filename in os.listdir(path4)[:1000]]
path5_files = [os.path.join(path5, filename) for filename in os.listdir(path5)[:1000]]

encoder = OneHotEncoder()
encoder.fit([[0], [1], [2], [3], [4]])

data = []
result = []

# Process images and labels
for path in path1_files:
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img)
    if img_array.shape == (128, 128, 3):
        data.append(img_array)
        result.append(encoder.transform([[0]]).toarray())

for path in path2_files:
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img)
    if img_array.shape == (128, 128, 3):
        data.append(img_array)
        result.append(encoder.transform([[1]]).toarray())

for path in path3_files:
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img)
    if img_array.shape == (128, 128, 3):
        data.append(img_array)
        result.append(encoder.transform([[2]]).toarray())

for path in path4_files:
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img)
    if img_array.shape == (128, 128, 3):
        data.append(img_array)
        result.append(encoder.transform([[3]]).toarray())

for path in path5_files:
    img = Image.open(path).resize((128, 128))
    img_array = np.array(img)
    if img_array.shape == (128, 128, 3):
        data.append(img_array)
        result.append(encoder.transform([[4]]).toarray())

data = np.array(data)
result = np.array(result)
result = result.reshape((len(data), 5))

x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.15, shuffle=True, random_state=42)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=15, batch_size=200, validation_data=(x_test, y_test))

# Save the model using the recommended Keras format
model.save('emotion_recognition_model.keras')
