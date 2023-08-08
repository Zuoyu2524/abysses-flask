import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import applications

from BaseModels import *

class Networks():
    def __init__(self, args):
        self.args = args
        if self.args.train_task == 'Semantic_Segmentation':
            if self.args.learning_model == 'Unet':
                print('Coming soon...')
                #self.learningmodel = Unet(self.args)
            if self.args.learning_model == 'DeepLab':
                print("Coming soon...")
        if self.args.train_task == 'Image_Classification':
            if self.args.backbone_name != 'None' and self.args.learning_model == 'CNN':
                self.learningmodel = CNN(self.args)

class CNN(Model):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        if self.args.pretrained_backbone:
            IMG_SHAPE = (self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels)
            if self.args.backbone_name == 'MobileNet':
                self.base_model = applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
                self.base_model.trainable = True
            if self.args.backbone_name == 'ResNet50':
                self.base_model = applications.resnet50(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
                self.base_model.trainable = True
            if self.args.backbone_name == 'Vgg16':
                self.base_model = applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
                self.base_model.trainable = True
        else:
            if 'mobilenet' in self.args.backbone_name:
                self.obj = MobileNet(self.args)
            if 'resnet' in self.args.backbone_name.lower() and 'v1' in self.args.backbone_name.lower():
                self.obj = ResNetV1(self.args)
            if 'resnet' in self.args.backbone_name.lower() and 'v2' in self.args.backbone_name.lower():
                self.obj = ResNetV2(self.args)
            if 'vgg' in self.args.backbone_name.lower():
                self.obj = Vgg(self.args)
            if 'xception' in self.args.backbone_name.lower():
                self.obj = Xception(self.args)

    def build_Model(self, input_data, reuse = False, name="CNN"):
        Layers = []
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            if self.args.pretrained_backbone:
                Layers.append(self.base_model(input_data))
            else:
                Layers = self.obj.build_Encoder_Layers(input_data, name = self.args.backbone_name)
            #Global Average Pooling 2D
            Layers.append(tf.reduce_mean(Layers[-1], axis = [1,2]))
            Layers.append(tf.layers.dropout(Layers[-1], 0.2, name= name + '_droput_1'))
            Layers.append(tf.layers.dense(Layers[-1], self.args.class_number, name = name + '_prediction'))
            if self.args.labels_type == 'onehot_labels':
                Layers.append(tf.nn.softmax(Layers[-1], name = name + '_softmax'))
            if self.args.labels_type == 'multiple_labels':
                Layers.append(tf.nn.sigmoid(Layers[-1], name = name + '_sigmoid'))

            return Layers
