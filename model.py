import csv
import os

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, MaxPooling2D
from keras.layers.convolutional import Conv2D


DATA_PATH = "./data2"
# DATA_PATH = "/opt/carnd_p3/data"
MODEL_NAME = "model_turns5.h5"
if not os.path.isfile(MODEL_NAME):
    LOAD_MODEL = None; SAVE_MODEL = MODEL_NAME
else:
    LOAD_MODEL = MODEL_NAME; SAVE_MODEL = "save_" + LOAD_MODEL
NUM_EPOCHS = 5

""" Import data and train """
lines = []
with open(DATA_PATH + "/driving_log.csv", 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
lines = lines[1:] # skip header line
print("Number of lines: ", len(lines))
print("LOAD_MODEL: {}, SAVE_MODEL: {}".format(LOAD_MODEL, SAVE_MODEL))

train_samples, valid_samples = train_test_split(lines, test_size=0.1)

def generator(samples, batch_size):
    N = len(samples)
    while True:
        sklearn.utils.shuffle(samples)
        steer_corr = 0.3
        steer_corr_factor = [steer_corr * n for n in [0, 1, -1]]    # [center, left, right]
        # Generate a single batch
        for offset in range(0, N, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_samples:
                for i in range(3):
                    fname = line[i].split('/')[-1] # source_path.split()[-1]
                    local_path = DATA_PATH + "/IMG/" + fname
                    img = cv2.imread(local_path)
                    img = img[70:-25, :]    # Crop out noisy scenery, keep only road terrain
                    img = cv2.resize(img, (32,32))  # Resize to avoid GPU memory issues
                    ang = float(line[3]) + steer_corr_factor[i]
                    # Add original image and ang
                    images.append(img)
                    angles.append(ang)
                    # Augment data with horizontally-flipped image (eq. to driving CCW)
                    images.append(np.fliplr(img))
                    angles.append(-ang)
            # Yield the batch
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Hyperparams
BATCH_SIZE = 32
NUM_BATCHES = np.ceil(len(train_samples)//BATCH_SIZE)

# compile and train the model using the generator function
train_gen = generator(train_samples, batch_size=BATCH_SIZE)
valid_gen = generator(valid_samples, batch_size=BATCH_SIZE)


""" Construct model """
def new_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(32,32,3)))   # Normalizing
    model.add(Conv2D(24,(5,5),activation='relu',strides=(2,2)))
    #model.add(Dropout(0.1))
    model.add(Conv2D(36,(5,5),activation='relu',strides=(2,2)))
    #model.add(Dropout(0.15))
    model.add(Conv2D(48,(5,5),activation='relu',strides=(2,2)))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64,(3,3),activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Conv2D(64,(3,3),activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="checkpoint",
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
print("Loading {}".format(LOAD_MODEL))
if LOAD_MODEL is None:
    model = new_model()
else:
    model = keras.models.load_model(LOAD_MODEL)
print("Number of lines: {} in {} batches".format(len(lines), NUM_BATCHES))
model.fit_generator(generator=train_gen,
                    steps_per_epoch=NUM_BATCHES,
                    validation_data=valid_gen,
                    validation_steps=NUM_BATCHES,
                    epochs=NUM_EPOCHS, verbose=1, callbacks=[model_checkpoint_callback])
print("Saving " + SAVE_MODEL)
model.save(SAVE_MODEL)
