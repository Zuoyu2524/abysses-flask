from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

import numpy as np
import pandas as pd
#import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.framework.python.ops import arg_scope

class MobileNet():
    def __init__(self, args):
        self.args = args
        self.bn_decay = 0.99
        self.bn_eps = 1e-3

    def build_Encoder_Layers(self, X, name = 'mobile_net'):
        chan_dim = -1
        Layers = []
        Layers.append(X)
        with tf.variable_scope(name):
            Layers.append(tf.layers.conv2d(Layers[-1], 32, (3 , 3), strides = 2, use_bias = False, padding = 'SAME', activation = None, kernel_initializer=tf.contrib.layers.xavier_initializer() ,name = name + '_convd_3x3_1'))
            Layers.append(tf.layers.batch_normalization(Layers[-1], axis = chan_dim, epsilon = self.bn_eps, momentum = self.bn_decay, name = name + '_bn_1'))
            Layers.append(tf.nn.relu(Layers[-1], name = name + '_relu_1'))

            #Mobile net blocks
            Layers.append(self.mobilenet_block(Layers[-1],  64, 1, name = name + '_block_1'))
            Layers.append(self.mobilenet_block(Layers[-1], 128, 2, name = name + '_block_2'))
            Layers.append(self.mobilenet_block(Layers[-1], 128, 1, name = name + '_block_3'))
            Layers.append(self.mobilenet_block(Layers[-1], 256, 2, name = name + '_block_4'))
            Layers.append(self.mobilenet_block(Layers[-1], 256, 1, name = name + '_block_5'))
            Layers.append(self.mobilenet_block(Layers[-1], 512, 2, name = name + '_block_6'))

            for i in range(5):
                Layers.append(self.mobilenet_block(Layers[-1], 512, 1, name = name + '_block_' + str(7 + i)))

            Layers.append(self.mobilenet_block(Layers[-1], 1024, 2, name = name + '_block_12'))
            Layers.append(self.mobilenet_block(Layers[-1], 1024, 1, name = name + '_block_13'))

            return Layers

    def mobilenet_block(self, X, filters, strides, name = 'block'):
        with tf.variable_scope(name):
            depthwise_filter = tf.get_variable('filters', (3,3, X.shape[-1],1), tf.float32)
            x =  tf.nn.depthwise_conv2d(X,filter=depthwise_filter, strides=[1,strides,strides,1], padding='SAME', name= name + 'depthwise_conv2d_3x3')
            x = tf.layers.batch_normalization(x, axis = -1, epsilon = self.bn_eps, momentum = self.bn_decay, name= name + 'bn_1')
            x = tf.nn.relu(x, name= name + 'relu_1')

            x = tf.layers.conv2d(x, filters, (1,1), use_bias=False, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name= name + 'conv2d_1x1')
            x = tf.layers.batch_normalization(x, axis = -1, epsilon = self.bn_eps, momentum = self.bn_decay, name= name + 'bn_2')
            x = tf.nn.relu(x, name= name + 'relu_2')

            return x

class ResNetV1():
    def __init__(self, args):
        super(ResNetV1, self).__init__()
        self.args = args
        self.filters = (64, 128, 256, 512)
        self.bn_eps = 2e-5
        self.bn_decay = 0.9
        if '18' in self.args.backbone_name:
            self.stages = (2, 2, 2, 2)
        if ('34' in self.args.backbone_name) or ('50' in self.args.backbone_name):
            self.stages = (3, 4, 6, 3)
        if '101' in self.args.backbone_name:
            self.stages = (3, 4, 23, 3)
        if '152' in self.args.backbone_name:
            self.stages = (3, 8, 36, 3)

    def build_Encoder_Layers(self, X, name = "ResnetV1"):

        Layers = []
        Layers.append(X)
        chan_dim = -1
        print(X)
        with tf.variable_scope(name):

            with tf.variable_scope('reduce_input_block'):
                # apply CONV => BN => ACT => POOL to reduce spatial size
                Layers.append(self.ZeroPadding(Layers[-1], 7, "ZeroPadding_1"))
                Layers.append(tf.layers.conv2d(Layers[-1], 64, 7,strides=(2, 2), use_bias=False, padding="valid", activation=None, name="conv2d_7x7"))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=self.bn_eps, momentum=self.bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))
                Layers.append(self.ZeroPadding(Layers[-1], 3, "ZeroPadding_2"))
                Layers.append(tf.layers.max_pooling2d(Layers[-1], (3, 3), strides=(2, 2), padding="valid", name = 'max_pooling2d'))

            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    if j == 0 and i != 0:
                        strides = (2 , 2)
                    else:
                        strides = (1 , 1)
                    if ('18' in self.args.backbone_name) or ('34' in self.args.backbone_name):
                        #Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        if j == 0 and i == 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        elif j == 0 and i != 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                    else:
                        #Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        if j == 0:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
            return Layers

    def Residual_Block_1(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            x = tf.layers.conv2d(X, filters, (3 , 3), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_3x3_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name= name + 'relu_1')

            x = tf.layers.conv2d(x, filters, (3, 3), strides= (1, 1), use_bias = False, padding="same", activation = None, name = name + 'conv2d_3x3_2')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")
                X = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')

            # add together the shortcut and the final CONV
            x = tf.nn.relu(tf.math.add(x, X, name = name + 'add_shortcut'), name = name + 'relu_2')

            return x

    def Residual_Block_2(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):

            x = tf.layers.conv2d(X, filters, (1, 1), strides=stride, use_bias = False, padding="same", activation = None, name = name + 'conv2d_1x1_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name='relu_1')

            x = tf.layers.conv2d(x, filters, (3, 3), strides=(1 , 1), use_bias = False, padding="same", activation = None, name= name + 'conv2d_3x3_1')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name=name + 'bn_2')
            x = tf.nn.relu(x, name = 'relu_2')

            x = tf.layers.conv2d(x, 4 * filters, (1, 1), strides=(1 , 1), use_bias = False, padding="same", activation = None, name= name + 'conv2d_1x1_2')
            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')



            if shorcut:
                X = tf.layers.conv2d(X, 4 * filters, (1, 1), strides = stride, use_bias = False, activation = None, name=name + "shorcut_conv2d_1x1")
                X = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'shorcut_bn_4')

            # add together the shortcut and the final CONV
            x = tf.nn.relu(tf.math.add(x, X, name = name + 'add_shortcut'), name = name + 'relu_3')

        # return the addition as the output of the ResNet module
        return x

    def ZeroPadding(self, X, Kernel_size = 3, name = "ZeroPadding"):
        p = int((Kernel_size - 1)/2)
        output = tf.pad(X, [[0, 0], [p, p], [p, p], [0, 0]], mode='CONSTANT' , constant_values=0, name= name)
        return output

class ResNetV2():
    def __init__(self, args):
        super(ResNetV2, self).__init__()
        self.args = args
        self.filters = (64, 128, 256, 512)
        self.bn_eps = 2e-5
        self.bn_decay = 0.9
        if '18' in self.args.backbone_name:
            self.stages = (2, 2, 2, 2)
        if ('34' in self.args.backbone_name) or ('50' in self.args.backbone_name):
            self.stages = (3, 4, 6, 3)
        if '101' in self.args.backbone_name:
            self.stages = (3, 4, 23, 3)
        if '152' in self.args.backbone_name:
            self.stages = (3, 8, 36, 3)

    def build_Encoder_Layers(self, X, name = "ResnetV2"):
        Layers = []
        Layers.append(X)
        chan_dim = -1
        with tf.variable_scope(name):

            with tf.variable_scope('reduce_input_block'):
                # apply CONV => BN => ACT => POOL to reduce spatial size
                Layers.append(self.ZeroPadding(Layers[-1], 7, "ZeroPadding_1"))
                Layers.append(tf.layers.conv2d(Layers[-1], 64, 7,strides=(2, 2), use_bias=False, padding="valid", activation=None, name="conv2d_7x7"))
                Layers.append(tf.layers.batch_normalization(Layers[-1], axis=chan_dim, epsilon=self.bn_eps, momentum=self.bn_decay, name='bn'))
                Layers.append(tf.nn.relu(Layers[-1], name='relu'))
                Layers.append(self.ZeroPadding(Layers[-1], 3, "ZeroPadding_2"))
                Layers.append(tf.layers.max_pooling2d(Layers[-1], (3, 3), strides=(2, 2), padding="valid", name = 'max_pooling2d'))

            for i in range(len(self.stages)):
                for j in range(self.stages[i]):
                    if j == 0 and i != 0:
                        strides = (2 , 2)
                    else:
                        strides = (1 , 1)
                    if ('18' in self.args.backbone_name) or ('34' in self.args.backbone_name):
                        #Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        if j == 0 and i == 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        elif j == 0 and i != 0:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_1(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                    else:
                        #Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        if j == 0:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = True, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
                        else:
                            Layers.append(self.Residual_Block_2(Layers[-1], self.filters[i], strides, chan_dim, shorcut = False, bn_eps=self.bn_eps, bn_decay = self.bn_decay, name = "block_" + str(i)+"_unit_" + str(j)))
            return Layers

    def Residual_Block_1(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            # the first block of the ResNet module are the 1x1 CONVs
            x = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name = name + 'relu_1')
            x = tf.layers.conv2d(x, filters, (3, 3), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_1x1_1')


            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, filters, (3, 3), strides=(1 , 1),  use_bias = False, padding="same", activation = None, name = name + 'conv2d_3x3_1')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")

            # add together the shortcut and the final CONV
            x = tf.math.add(x, X, name = name + 'add_shortcut')

            return x

    def Residual_Block_2(self, X, filters = 64, stride = 2, chan_dim = -1, shorcut = False, bn_eps = 2e-5, bn_decay = 0.9, name = "block1"):
        """
        data: input to the residual module
        filters: number of filters that will be learned by the final CONV layer (the first two CONV layers will learn filters/4 filters)
        stride: controls the stride of the convolution (will help us reduce spatial dimensions without using max pooling)
        chanDim: defines the axis which will perform batch normalization (-1)
        red (i.e. reduce) will control whether we are reducing spatial dimensions (True) or not (False) as not all residual modules will reduce dimensions of our spatial volume
        bnEps: controls the Ɛ responsible for avoiding “division by zero” errors when normalizing inputs
        bnMom: controls the momentum for the moving average
        """
        with tf.variable_scope(name):
            x = tf.layers.batch_normalization(X, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_1')
            x = tf.nn.relu(x, name = name + 'relu_1')
            x = tf.layers.conv2d(x, filters, (1, 1), strides=stride, use_bias = False, padding = "same",  activation = None, name = name + 'conv2d_1x1_1')

            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_2')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, filters, (3, 3), strides = (1 , 1), use_bias = False, padding = "same", activation = None, name = name + 'conv2d_3x3_2')

            x = tf.layers.batch_normalization(x, axis = chan_dim, epsilon = bn_eps, momentum = bn_decay, name = name + 'bn_3')
            x = tf.nn.relu(x, name = name + 'relu_2')
            x = tf.layers.conv2d(x, 4 * filters, (1, 1), strides = (1 , 1), use_bias = False, padding = "same", activation = None, name = name + 'conv2d_1x1_3')

            # if we are to reduce the spatial size, apply a CONV layer to the shortcut
            if shorcut:
                X = tf.layers.conv2d(X, 4 * filters, (1, 1), strides = stride, use_bias = False, activation = None, name = name + "reduce_conv2d_1x1")

            # add together the shortcut and the final CONV
            x = tf.math.add(x, X, name = name + 'add_shortcut')

        # return the addition as the output of the ResNet module
        return x

    def ZeroPadding(self, X, Kernel_size = 3, name = "ZeroPadding"):
        p = int((Kernel_size - 1)/2)
        output = tf.pad(X, [[0, 0], [p, p], [p, p], [0, 0]], mode='CONSTANT' , constant_values=0, name= name)
        return output

class Xception():
    def __init__(self, args):
        self.args = args

    def build_Encoder_Layers(self, X, name = 'xception'):
        Layers = []
        Layers.append(X)

        with tf.variable_scope(name):
            Layers.append(self.general_conv2d(Layers[-1], 32, 3, stride=2, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_1'))
            Layers.append(self.general_conv2d(Layers[-1], 64, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_2'))
            tensor = Layers[-1]

            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_3'))
            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_4'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_1'))


            tensor = self.general_conv2d(tensor, 128, 1, stride=2, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_5')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_1'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_1'))

            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_6'))
            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_7'))
            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_1'))


            tensor = self.general_conv2d(tensor, 256, 1, stride=2, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_8')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_2'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_2'))


            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_9'))
            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_10'))
            #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_2'))


            tensor = self.general_conv2d(tensor, 728, 1, stride=1, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_11')
            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_3'))
            Layers.append(tf.nn.relu(Layers[-1], name='relu_3'))

            #Middle flow
            for i in range(8):
                tensor = Layers[-1]
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_1' + str(i)))
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_2' + str(i)))
                Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                                  padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_12_3' + str(i)))
                Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_4' + str(i)))

            Layers.append(tf.nn.relu(Layers[-1], name='relu_4'))
            #Exit flow
            tensor = self.general_conv2d(Layers[-1], 1024, 1, stride=1, conv_type = 'conv',
                                               padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_13')

            Layers.append(self.general_conv2d(Layers[-1], 728, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_14'))
            Layers.append(self.general_conv2d(Layers[-1], 1024, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_15'))
            #Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_3'))

            Layers.append(tf.math.add(tensor, Layers[-1], name = 'shorcut_4'))

            Layers.append(self.general_conv2d(Layers[-1], 1536, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='relu', do_norm=True, name=name + '_conv2d_16'))
            Layers.append(self.general_conv2d(Layers[-1], 2048, 3, stride=1, conv_type = 'dep_conv',
                                              padding='SAME', activation_function='None', do_norm=True, name=name + '_conv2d_17'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, conv_type = 'conv', stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            if conv_type == 'conv':
                conv = tf.layers.conv2d(input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            if conv_type == 'dep_conv':
                conv = tf.contrib.layers.separable_conv2d(input_data, filters, kernel_size, 1, stride, padding, activation_fn = None, weights_initializer = tf.contrib.layers.xavier_initializer())

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv

class Vgg():
    def __init__(self, args):
        self.args = args

    def build_Encoder_Layers(self, X, name = 'vgg'):
        Layers = []
        Layers.append(X)

        with tf.variable_scope(name):
            Layers.append(self.general_conv2d(Layers[-1], 64, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_1'))
            Layers.append(self.general_conv2d(Layers[-1], 64, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_2'))

            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_1'))


            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_3'))
            Layers.append(self.general_conv2d(Layers[-1], 128, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_4'))

            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_2'))


            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_5'))
            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_6'))
            Layers.append(self.general_conv2d(Layers[-1], 256, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_7'))

            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_3'))


            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_8'))
            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_9'))
            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_10'))

            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_4'))


            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_11'))
            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_12'))
            Layers.append(self.general_conv2d(Layers[-1], 512, 3, stride=1, conv_type = 'conv',
                                              padding='SAME', activation_function='relu', do_norm=False, name=name + '_conv2d_13'))

            Layers.append(tf.layers.max_pooling2d(Layers[-1], 2, 2, name=name + '_maxpooling_5'))

            return Layers

    def general_conv2d(self, input_data, filters=64,  kernel_size=7, stride=1, conv_type = 'conv', stddev=0.02, activation_function="relu", padding="VALID", do_norm=True, relu_factor=0, name="conv2d"):
        with tf.variable_scope(name):
            if conv_type == 'conv':
                conv = tf.layers.conv2d(input_data, filters, kernel_size, stride, padding, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
            if conv_type == 'dep_conv':
                conv = tf.contrib.layers.separable_conv2d(input_data, filters, kernel_size, 1, stride, padding, activation_fn = None, weights_initializer = tf.contrib.layers.xavier_initializer())

            if do_norm:
                conv = tf.layers.batch_normalization(conv, momentum=0.9)

            if activation_function == "relu":
                conv = tf.nn.relu(conv, name='relu')
            if activation_function == "leakyrelu":
                conv = tf.nn.leaky_relu(conv, alpha=relu_factor)
            if activation_function == "elu":
                conv = tf.nn.elu(conv, name='elu')

            return conv
