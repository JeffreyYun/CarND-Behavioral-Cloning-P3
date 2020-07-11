import csv
import cv2
import numpy as np
import keras
from keras.model import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


""" Import data and train """
lines = []
with open("../data/driving_log.csv", 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

images = []
measurements = []
steer_correction = 0.2
steer_corr_factor = [0, 1, -1]    # [center, left, right]
for line in lines:
    for i in range(3):
        source_path = line[0]
        tokens = source_path.split('/')
        fname = tokens[-1]
        local_path = f"../data/IMG/{fname}"
        image = cv2.imread(local_path)
        measurement = float(line[3]) + steer_corr_factor[i] * steer_correction
        # Add image and measurement
        images.append(image)
        measurements.append(measurement)
        # Augment data with horizontally-flipped image (eq. to driving CCW)
        images.append(np.fliplr(image))
        measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)


""" Construct model """
# TODO: Experiment more with model architecture
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))   # Normalizing
model.add(Cropping2D(cropping=((70,25), (0,0))))    # Cropping (out the top noisy scenery)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# TODO: Use generators to reduce memory usage from data


# TODO: Collect more data using your own driving (GPU)


# TODO: Eat dinner while training
