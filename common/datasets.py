import gzip
import numpy as np

def load_data_from_gz(dataset):

    dataset_root = '/home/zxz/dongshuai/data/Cluster/'+dataset

    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(dataset_root+'/'+fname)

    with gzip.open(paths[0], 'rb') as lbpath:
        # Interpret a buffer as a 1-dimensional array.
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)

def load_mnist():
    # the data, shuffled and split between train and test sets
#    from tensorflow.keras.datasets import mnist
#    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_data_from_gz('mnist')
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 1, 28, 28]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_mnist_test():
    # the data, shuffled and split between train and test sets
#    from tensorflow.keras.datasets import mnist
#    _, (x, y) = mnist.load_data()
    _, (x, y) = load_data_from_gz('mnist')
    x = x.reshape([-1, 1, 28, 28]) / 255.0
    print('MNIST samples', x.shape)
    return x, y


def load_fashion_mnist():
#    from tensorflow.keras.datasets import fashion_mnist  # this requires keras>=2.0.9
#    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_data_from_gz('fashion-mnist')
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape([-1, 1, 28, 28]) / 255.0
    print('Fashion MNIST samples', x.shape)
    return x, y


def load_usps(data_path='./data/usps'):
    import os
    if not os.path.exists(data_path+'/usps_train.jf'):
        raise ValueError("No data for usps found, please download the data from links in \"./data/usps/download_usps.txt\".")

    with open(data_path + '/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open(data_path + '/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64') / 2.
    y = np.concatenate((labels_train, labels_test))
    x = x.reshape([-1, 16, 16, 1])
    print('USPS samples', x.shape)
    return x, y


def load_data_conv(dataset):
    if dataset == 'mnist':
        return load_mnist()
    elif dataset == 'mnist-test':
        return load_mnist_test()
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist()
    elif dataset == 'usps':
        return load_usps()
    else:
        raise ValueError('Not defined for loading %s' % dataset)


def load_data(dataset):
    x, y = load_data_conv(dataset)
    return x.reshape([x.shape[0], -1]), y

