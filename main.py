import numpy as np
from matchingclassifier.MatchingClassifier1D import MatchingClassifier1D
from sklearn import metrics
from keras.datasets import mnist
from keras import backend as K
import keras

# X = np.random.rand(100)
# X = np.concatenate((X, np.random.rand(100)+2), axis=0)
# X = np.concatenate((X, np.random.rand(100)+5), axis=0)
#
# c = np.zeros(100)
# c = np.concatenate((c, np.ones(100)), axis=0)
# c = np.concatenate((c, np.ones(100)*2), axis=0)
#
# X = np.reshape(X, (-1, 1))
# c = np.reshape(c, (-1, 1))
#
# mc = MatchingClassifier1D()
# mc.fit_1D(X, c)

# y = mc.predict(X)
# print(metrics.accuracy_score(y, c))

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train[0:1000, :, :, :]
y_train = y_train[0:1000]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

