import keras
import numpy as np
import json
import os
from keras.models import  load_model
from data.data_gen import DataGenerator
from   complexnn                             import ComplexBN,\
                                                    ComplexConv1D,\
                                                    ComplexConv2D
import  tensorflow as tf
#test the result good or bad

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

file_path  = '/home/zzzj/PycharmProjects/keras-complex/data/autoGAN'
# img_path = '/home/zzzj/PycharmProjects/pytorch-CycleGAN-and-pix2pix/results/apple2orange_pretrained/test_latest/images'
# img_path = '/home/zzzj/PycharmProjects/pytorch-CycleGAN-and-pix2pix/results/summer2winter_yosemite_pretrained/test_latest/images'
img_path = '/home/zzzj/PycharmProjects/keras-complex/data/images_apple'

# model_path ='/home/zzzj/PycharmProjects/keras-complex/results4/chkptsOriginal'
model_path ='/home/zzzj/PycharmProjects/keras-complex/resultAutoGan/best'


batch_size = 8
model = load_model(os.path.join(model_path, 'Bestmodel_000001_0.5593_0.7833.hdf5'),custom_objects={'ComplexConv2D': ComplexConv2D, 'ComplexBatchNormalization': ComplexBN})

# partion = json.load(open(os.path.join(file_path,'only_test.json')))
# labels  = json.load(open(os.path.join(file_path, 'test_labels.json')))

partion = json.load(open(os.path.join(file_path, 'result.json')))
labels = json.load(open(os.path.join(file_path, 'labels.json')))

test_generator = DataGenerator(partion['test'],  labels, data_root=img_path, batch_size=batch_size, shuffle=False,mode='test')
test_loss, test_acc = model.evaluate_generator(test_generator, verbose=1)
print('test_loss: {}, test_acc: {}'.format(test_loss, test_acc))


# img_path =  '/home/zzzj/下载/images/001000-20200430T080854Z-001/001000'
# partion  = json.load(open(os.path.join(file_path,'StyleGAN.json')))
# labels   = json.load(open(os.path.join(file_path,'StyleGAN_labels.json')))
# test_generator = DataGenerator(partion['test'], labels, data_root=img_path,batch_size=batch_size)
# # acc = model.predict_generator(test_generator,verbose=1)
# test_loss, test_acc = model.evaluate_generator(test_generator,verbose=1)
# print('test_loss: {}, test_acc: {}'.format(test_loss, test_acc))
