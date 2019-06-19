"""Darknet-53 for yolo v3.
"""

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Reshape, Activation, GlobalAveragePooling2D, Dense
from keras.layers import add, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K


def conv2d_unit(x, filters, kernels, strides=1):
    """Convolution Unit
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
            Output tensor.
    """
    x = Conv2D(filters, kernels,
               padding='same',
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filters):
    """Residual Block
    This function defines a 2D convolution operation with BN and LeakyReLU.

    # Arguments
        x: Tensor, input tensor of residual block.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.

    # Returns
        Output tensor.
    """
    x = conv2d_unit(inputs, filters, (1, 1))
    x = conv2d_unit(x, 2 * filters, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)

    return x


def stack_residual_block(inputs, filters, n):
    """Stacked residual Block
    """
    x = residual_block(inputs, filters)

    for i in range(n - 1):
        x = residual_block(x, filters)

    return x


def darknet_base(inputs):
    """Darknet-53 base model.
    """

    x = conv2d_unit(inputs, 32, (3, 3))

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 256, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x


def darknet53_model(img_rows, img_cols, color_type=1, num_classes=None, fine_tuning=True, load_weights=True):
    """Darknet-53 classifier.
    """
    img_input = Input(shape=(img_rows, img_cols, color_type))
    x = darknet_base(img_input)

    x_fc = GlobalAveragePooling2D()(x)
    x_fc = Dense(1000, activation='softmax')(x)

    model = Model(img_input, x_fc)
    
    if load_weights:
        # Load ImageNet pre-trained data 
        if K.image_dim_ordering() == 'th':
          # Use pre-trained weights for Theano backend
          weights_path = './cnn_finetune/imagenet_models/darknet53_weights_th_dim_ordering_th_kernels.h5'
        else:
          # Use pre-trained weights for Tensorflow backend
          weights_path = './cnn_finetune/imagenet_models/darknet53_weights_tf_dim_ordering_tf_kernels.h5'

        model.load_weights(weights_path)

    if fine_tuning:
        # Truncate and replace softmax layer for transfer learning
        # Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model
        x_newfc = GlobalAveragePooling2D(name='avg_pool')(x)
        x_newfc = Dense(num_classes, activation='softmax', name='fc'+str(num_classes))(x_newfc)

        # Create another model with our customized softmax
        model = Model(img_input, x_newfc)
        
    print('darknet53_model:', 'classes:', num_classes, 'fine-tuning:', fine_tuning)    
    model.summary()
    print('-'*100)

    return model


if __name__ == '__main__':
    model = darknet()
    print(model.summary())