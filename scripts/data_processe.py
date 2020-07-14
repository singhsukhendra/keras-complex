from  tensorflow.keras.datasets import cifar10,mnist
import numpy as np
import argparse
def normalize(data):
    return (data-data.min())/data.max()

def normalize2(complex_data):
    data = complex_data - complex(complex_data.real.min(), complex_data.imag.min())
    return data/np.abs(complex_data).max()


def load_data(dataName):
    if dataName == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    n_train = 45000
    random_id = np.arange(len(x_train))
    np.random.seed(0xDEADBEEF)
    np.random.shuffle(random_id)
    train_id = random_id[:n_train]
    val_id = random_id[n_train:]

    X_train = x_train[train_id]
    X_val = x_train[val_id]
    Y_train = y_train[train_id]
    Y_val = y_train[val_id]

    return X_train, X_val, x_test, Y_train, Y_val, y_test


# transform colour image to gray image
# def fft_norm(data):
#     if data.ndim == 4:
#         data = 0.2126 * data[:,:,:,0] + 0.7152 * data[:,:,:,1] + 0.0722 * data[:,:,:,2]
#     data = np.expand_dims(data,axis=-1)
#     print(data.shape)
#     # f = np.fft.fft2(data)
#     f = np.fft.fftn(data,axes=(1,2))
#     f = f/(data.shape[1]*data.shape[2])
#     real_value = f.real; imag_value = f.imag
#     print('real max is {}, real min is {},imag max is {}, imag min is {}'.format(real_value.max(),real_value.min(),imag_value.max(), imag_value.min()))
#     result = np.concatenate([real_value, imag_value], axis=-1)
#     return result
def fft_norm(data):
    # f = np.fft.fftshift(np.fft.fft2(data))
    if data.shape ==3:
        data = np.expand_dims(data, axis=-1)
    # data = data.astype('float32')/255.
    f = np.fft.fftn(data, axes=(1,2))
    # f = f/(np.abs(f)+1e-10)# exist 0 + 0j
    f = f/(data.shape[1]*data.shape[2])
    magnitude = np.log(1e-3+np.abs(f)) # 1? 1e-3?
    angle     = np.angle(f)
    # real_value = f.real; imag_value = f.imag
    real_value = magnitude;   imag_value = angle
    print('real max is {}, real min is {},imag max is {}, imag min is {}'.format(real_value.max(),real_value.min(),imag_value.max(), imag_value.min()))




    result   = np.concatenate([real_value, imag_value], axis=-1)
    # result = np.stack([f.real, f.imag],axis=3)# shape:(10000,32,32,2,3)
    print(result.shape)
    return result


if __name__ =='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--data', default='cifar10')
    args = ap.parse_args()
    X_train, X_val, x_test, Y_train, Y_val, y_test = load_data(args.data)

    data_train = fft_norm(X_train)
    data_test = fft_norm(x_test)
    data_val = fft_norm(X_val)

    if args.data == 'mnist':
        data_path = './fft_data/'
    else:
        data_path = './fft_cifar_data/'

    np.save(data_path+'train_data.npy', data_train)
    np.save(data_path+'test_data.npy', data_test)
    np.save(data_path+'val_data.npy', data_val)
    np.save(data_path+'train_y.npy', Y_train)
    np.save(data_path+'val_y.npy', Y_val)
    np.save(data_path+'test_y.npy', y_test)

    print('process successfully!')