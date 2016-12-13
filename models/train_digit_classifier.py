import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

import cv2
from imutils import paths


def prepare_image(image, img_rows, img_cols):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (img_rows, img_cols))
	image = image.reshape((1, img_rows, img_cols))
	return image


def build_batch(im_path, img_rows, img_cols):
	# load the images from disk, prepare them for extraction, and convert
	# the list to a NumPy array
	images = [prepare_image(cv2.imread(p), img_rows, img_cols) for p in im_path]

	# extract the labels from the image dir
	labels = [p.split("/")[-2] for p in im_path]

	# return the labels and images
	return np.asarray(images), np.asarray(labels)


batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.empty((0, 1, img_rows, img_cols))
y_train = np.empty((0))
X_test = np.empty((0, 1, img_rows, img_cols))
y_test = np.empty((0))
for i in range(0, 10):
	(X, Y) = build_batch(sorted(list(paths.list_images("dataset/{}".format(i)))), img_rows, img_cols)
	X_train = np.concatenate((np.asarray(X[0:75]), X_train))
	y_train = np.concatenate((np.asarray(Y[0:75]), y_train))
	X_test = np.concatenate((np.asarray(X[75:]), X_test))
	y_test = np.concatenate((np.asarray(Y[75:]), y_test))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

if K.image_dim_ordering() == 'th':
	X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('digit.h5')
print("model saved")
