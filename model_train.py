import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128

def load_data():
    data = []
    labels = []

    for label, folder in enumerate(['clean', 'stego']):
        folder_path = f'data/{folder}'
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)

    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    labels = to_categorical(labels, 2)
    return train_test_split(data, labels, test_size=0.2)

X_train, X_test, y_train, y_test = load_data()

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)
model.save("steganalysis_model.h5")
