from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

np.random.seed(1337)

weights = R'weights/weights-improvement-{epoch:02d}-{val_loss:.4f}-LargerSeed.hdf5'
batch_size = 100
epochs = 500
num_classes = 10

(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

# Data is loaded as integers, so it must be cast to floating points to perform division.
train_features = train_features.astype(float)/255
test_features = test_features.astype(float)/255

# Converts a class vector to binary class matrix
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=train_features.shape[1:]))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=1, mode='min')
checkpoint = ModelCheckpoint(weights, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callback_list = [early_stopping, checkpoint]

model.fit_generator(datagen.flow(train_features, train_labels,
                    batch_size=batch_size),
                    epochs=epochs,
                    callbacks=callback_list,
                    validation_data=(test_features, test_labels),
                    verbose=1)

model.save('cifar10_model_v2.h5')

