# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed

from keras import backend as K

from keras_frcnn.RoiPoolingConv import RoiPoolingConv
from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization

import numpy as np
import keras
import pickle

from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

def get_weight_path():
    if K.image_data_format() == 'channels_first':
        return 'densenet_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'densenet_weights_tf_dim_ordering_tf_kernels.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height) 

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(x, nb_filter, dropout_rate=None, name=None):
    
    inter_channel = nb_filter*4

    # 1x1 convolution
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name+'_bn1')(x)
    x = Activation('relu', name=name+'_relu1')(x)
    x = Conv2D(inter_channel, 1, 1, name=name+'_conv1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 convolution
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name+'_bn2')(x)
    x = Activation('relu', name=name+'_relu2')(x)
    x = ZeroPadding2D((1, 1), name=name+'_zeropadding2')(x)
    x = Conv2D(nb_filter, 3, 1, name=name+'_conv2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, 
                grow_nb_filters=True, name =None):

    concat_feat = x # store the last layer output

    for i in range(nb_layers):
        
        branch = i+1
        x =conv_block(concat_feat, growth_rate, dropout_rate, name=name+str(stage)+'_block'+str(branch)) # 在参考的基础，修改的地方这里应该是相同的growth_rate=32
        concat_feat = Concatenate(axis=3, name=name+str(stage)+'_block'+str(branch))([concat_feat, x])

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def transition_block (x,stage, nb_filter, compression=1.0, dropout_rate=None, name=None):

    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name+str(stage)+'_bn')(x)
    x = Activation('relu', name=name+str(stage)+'_relu')(x)
    
    x = Conv2D(int(nb_filter*compression), 1, 1, name=name+str(stage)+'_conv', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    
    x = AveragePooling2D((2,2), strides=(2,2), name=name+str(stage)+'_pooling2d')(x)

    return x


# def nn_base(input_tensor=None, trainable=False):
    if K.image_data_format()=='channels_first':
        input_shape=(3, None, None)
    else:
        input_shape=(None, None, 3)

    if input_tensor is None:
        img_input=Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input=Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input=input_tensor
    
    if K.image_data_format()=='channels_last':
        bn_axis=3
    else:
        bn_axis=1

    
    growth_rate = 12
    inpt = Input(shape=(32,32,3))
    x = Conv2D(growth_rate*2, (3, 3), strides=1, padding='same')(inpt)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
    x = TransitionLayer(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = BatchNormalization(axis=3)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    return x

def nn_base(input_tensor=None, trainable=False, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
             classes=1000, weights_path=None):
    compression = 1.0 - reduction
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121
    if K.image_data_format()=='channels_first':
        input_shape=(3, None, None)
    else:
        input_shape=(None, None, 3)

    if input_tensor is None:
        img_input=Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input=Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input=input_tensor
    
    if K.image_data_format()=='channels_last':
        bn_axis=3
    else:
        bn_axis=1

    # img_input = Input(shape=(224, 224, 3))

    # initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name='conv1_bn')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Dense block and Transition layer
    for block_id in range(nb_dense_block - 1):
        stage = block_id + 2  # start from 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_id], nb_filter, growth_rate,
                                   dropout_rate=dropout_rate, name='Dense')

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, name='Trans')
        nb_filter *= compression

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
                               dropout_rate=dropout_rate, name='Dense')

    # top layer
    # x = BatchNormalization(name='final_conv_bn')(x)
    # x = Activation('relu', name='final_act')(x)
    # x = GlobalAveragePooling2D(name='final_pooling')(x)
    # x = Dense(classes, activation='softmax', name='fc')(x)
    return x


def rpn(base_layers, num_anchors):

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 14
        input_shape = (num_rois, 14, 14, 512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois, 512, 7, 7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(BatchNormalization(name='final_conv_bn'))(out_roi_pool)
    out = TimeDistributed(Activation('relu', name='final_act'))(out)
    out = TimeDistributed(GlobalAveragePooling2D(name='final_pooling'))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]