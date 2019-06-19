import cv2
import numpy as np

from keras.datasets import cifar10, cifar100
from keras import backend as K
from keras.utils import np_utils

# nb_train_samples = 3000 # 3000 training samples
# nb_valid_samples = 100 # 100 validation samples
# num_classes = 10

def load_cifar10_data(img_rows, img_cols, num_classes=10, nb_train_samples=50000, nb_valid_samples=10000):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()
    print('cifar10 trainset:', type(X_train), X_train.shape, type(Y_train), Y_train.shape)
    print('cifar10 valset:', type(X_valid), X_valid.shape, type(Y_valid), Y_valid.shape)
    print('-'*100)

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid


def load_cifar100_data(img_rows, img_cols, num_classes=100, nb_train_samples=50000, nb_valid_samples=10000):

    # Load cifar100 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar100.load_data()
    print('cifar100 trainset:', type(X_train), X_train.shape, type(Y_train), Y_train.shape)
    print('cifar100 valset:', type(X_valid), X_valid.shape, type(Y_valid), Y_valid.shape)
    print('-'*100)

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid