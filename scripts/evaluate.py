#coding:utf-8
import keras
import keras.models as KM
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.models import model_from_yaml
from complexnn import ComplexConv2D,ComplexBN,GetReal,GetImag
import sys
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

#1.HDF5格式文件保存的是 ： Model weights
# stucture2.H5 格式文件保存的是： Model stucture 和 Model weights
# 3.JSON 和 YAML 格式问价保存的是： Model

def fft_norm(data, path):
    # f = np.fft.fftshift(np.fft.fft2(data))
    if data.shape ==3:
        data = np.expand_dims(data, axis=-1)

    f = np.fft.fftn(data, axes=(1,2))
    # f = f/(np.abs(f)+1e-10)# exist 0 + 0j
    f = f/(data.shape[1]*data.shape[2])
    # max_value = max(f.real.max(),f.imag.max()); min_value = min(f.real.min(), f.imag.min())
    # real_value = (f.real - min_value)/(max_value - min_value)
    # imag_value = (f.imag - min_value)/(max_value - min_value)
    real_value = f.real
    imag_value = f.imag
    # print('real max is {}, real min is {},imag max is {}, imag min is {}'.format(f.real.max(),f.real.min(),f.imag.max(), f.imag.min()))
    print('real max is {}, real min is {},imag max is {}, imag min is {}'.format(real_value.max(),real_value.min(),imag_value.max(), imag_value.min()))
    if data.shape[-1] == 1:

        result = np.concatenate([real_value, imag_value], axis=-1)
    else:
        # ch1 = np.concatenate([np.expand_dims(f[:,:,:,0].real, axis=-1),np.expand_dims(f[:,:,:,0].imag, axis=-1)], axis=-1)
        # ch2 = np.concatenate([np.expand_dims(f[:,:,:,1].real, axis=-1),np.expand_dims(f[:,:,:,1].imag, axis=-1)],axis=-1)
        # ch3 = np.concatenate([np.expand_dims(f[:,:,:,2].real, axis=-1),np.expand_dims(f[:,:,:,2].imag, axis=-1)],axis=-1)
        # result   = np.concatenate([ch1, ch2, ch3], axis=-1)
        result   = np.concatenate([real_value, imag_value], axis=-1)
    # result = np.stack([f.real, f.imag],axis=3)# shape:(10000,32,32,2,3)
    print(result.shape)
    np.save(os.path.join(path,'cifar10'), result)
    return result

def load_data(path):
    imglist = os.listdir(path)
    imgNum = len(imglist)
    data = np.empty((imgNum,32,32,3), dtype='float32')
    if not os.path.exists(path):
        print('path is not exist')
    for i in range(imgNum):
        img = cv2.imread(os.path.join(path,imglist[i])).astype('float32') #divide 255?
        data[i] = img
    np.save('./results/cifar_norm/fake_data/cifar10', data)
    return data

structure_path = './results/cifar_norm/best/Bestmodel_000186_0.7944_0.7706.yaml'
# structure_path = '/home/zzzj/PycharmProjects/keras-complex/scripts/results/cifar_norm/chkptsOriginal/ModelChkpt000200.yaml'
chkptFilename = './results/cifar_norm/best/Bestmodel_000186_0.7944_0.7706.hdf5'

# model = model_from_yaml(structure_path)
# model.load_weights(, by_name=True)

model = KM.load_model(chkptFilename, custom_objects={
    "ComplexConv2D": ComplexConv2D,
    "ComplexBatchNormalization": ComplexBN,
    "GetReal": GetReal,
    "GetImag": GetImag
})
# print(model.summary())
oridata_path = '/home/zzzj/PycharmProjects/GAN/scripts/GeneratedFakeImg'
data_path = '/home/zzzj/PycharmProjects/keras-complex/scripts/results/cifar_norm/fake_data'
fake_data = load_data(oridata_path)
# fake_data = np.load(os.path.join(data_path,'cifar10.npy'))
data = fft_norm(fake_data, data_path)

print(data.shape)
results = np.argmax(model.predict(data), axis=1)
print(results)
print((results==6).sum()/float(len(results)))