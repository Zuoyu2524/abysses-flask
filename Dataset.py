import os
import sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

from PIL import Image
from Tools import *

class Dataset():
    def __init__(self, args):
        self.file_paths = []
        self.args = args

        if 'SW_fragments' in self.args.criteria:
            self.class_names = ['0-10%','10-50%','50-100%']
        if 'lithology' in self.args.criteria:
            self.class_names = ['Slab','Sulfurs','Vocanoclastic']
        if 'morphology' in self.args.criteria:
            self.class_names = ['Fractured','Marbled','ScreeRubbles', 'Sedimented']

        self.class_number = len(self.class_names)
        #Looking for all images
        files = os.listdir(self.args.dataset_main_path)
        for file in files:
            self.file_paths.append(self.args.dataset_main_path + file)

    def read_samples(self, image_paths):
        # 1. Read image and labels
        images = np.zeros((self.args.batch_size, self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels))

        for i in range(len(image_paths)):

            image = Image.open(image_paths[i])
            if self.args.resize:
                newsize = (self.args.new_size_rows, self.args.new_size_cols)
                images[i, :, :, :] = self.preprocess_input(np.array(image.resize(newsize, Image.ANTIALIAS), dtype = np.float32), "None")
            else:
                images[i, :, :, :] = self.preprocess_input(np.array(image, dtype = np.float32), "None")
        return images

    def preprocess_input(self, data, backbone_name):
        if backbone_name == 'movilenet':
            data = tf.keras.applications.mobilenet.preprocess_input(data)
        elif backbone_name == 'resnet50':
            data = tf.keras.applications.resnet.preprocess_input(data)
        elif backbone_name == 'vgg16':
            data = tf.keras.applications.vgg16.preprocess_input(data)
        else:
            data = data/255.

        return data
