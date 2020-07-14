import complexnn
# import keras
# from keras import models
# from keras import layers
# from keras import optimizers
# from keras.datasets import mnist,cifar10
# from keras import backend as K

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.datasets import mnist,cifar10
from keras import backend as K
from keras.regularizers import l2
import tensorflow.compat.v1 as tf
import numpy as np
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0,1'



batch_size = 128
num_classes = 10
epochs = 100

data_type = 'cifar10'
model_type = 'complex'
img_rows, img_cols = (28,28) if data_type == 'mnist' else (32,32)
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist_train = tfd.load('mnist', )
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format()== 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

if model_type == 'complex':
    # fft_train = np.fft.fft2(x_train)
    # fft_test = np.fft.fft2(x_test)
    # train_real = normalize(np.real(fft_train)); train_imag = normalize(np.imag(fft_train))
    # test_real = normalize(np.real(fft_test));   test_imag  = normalize(np.imag(fft_test))
    train_real = np.load('./fft_data/train_real2.npy'); train_imag = np.load('./fft_data/train_imag2.npy')
    test_real = np.load('./fft_data/test_real2.npy'); test_imag = np.load('./fft_data/test_imag2.npy')
    x_train = np.concatenate([train_real, train_imag], axis=-1)
    x_test  = np.concatenate([test_real,  test_imag],axis=-1)
    input_shape = (img_rows, img_cols, 3*2)
print(x_train.shape, y_train.shape, 'train_samples', 'train_label_samples')
print(x_test.shape, y_test.shape, 'test_samples', 'test_label_samples')
print(y_test.max())
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('input_shape:', input_shape)
model = models.Sequential()
# complex value network
if model_type == 'complex':
    model.add(complexnn.conv.ComplexConv2D(32,(3,3),activation='relu', input_shape=input_shape))
    model.add(complexnn.bn.ComplexBatchNormalization())
    model.add(layers.MaxPool2D((2,2),padding='same'))
    model.add(layers.Dropout(.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
else:
# real value network
    model.add(layers.convolutional.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.Dropout(.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test Accuracy', score[1])