import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.model import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D


""" Import data and train """
lines = []
with open("../data/driving_log.csv", 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)

train_samples, valid_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    N = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        # Generate a single batch
        for offset in range(0, N, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            steer_corr = 0.2
            steer_corr_factor = [steer_corr * n for n in [0, 1, -1]]    # [center, left, right]
            for line in batch_samples:
                for i in range(3):
                    fname = line[0].split('/')[-1] # source_path.split()[-1]
                    local_path = f"../data/IMG/{fname}"
                    image = cv2.imread(local_path)
                    measurement = float(line[3]) + steer_corr_factor[i]
                    # Add original image and measurement
                    images.append(image)
                    angles.append(measurement)
                    # Augment data with horizontally-flipped image (eq. to driving CCW)
                    images.append(np.fliplr(image))
                    angles.append(-measurement)
                # Yield the batch
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

# Hyperparams
BATCH_SIZE = 32

# compile and train the model using the generator function
train_gen = generator(train_samples, batch_size=BATCH_SIZE)
valid_gen = generator(valid_samples, batch_size=BATCH_SIZE)


""" Construct model """
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))   # Normalizing
model.add(Cropping2D(cropping=((70,25), (0,0))))    # Cropping (out the top noisy scenery)
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.15))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Dropout(0.2))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64,5,5,activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator=train_gen,
                    steps_per_epoch=np.ceil(len(train_samples)/BATCH_SIZE),
                    validation_data=valid_gen,
                    validation_steps=np.ceil(len(valid_samples)/BATCH_SIZE),
                    epochs=5, verbose=1)
model.save('model.h5')
