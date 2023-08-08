import os
import sys
import json
import argparse
import numpy as np
from glob import glob
from datetime import datetime

from Model import *
from Dataset import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_task', dest='train_task', type=str, default='Image_Classification', help='Learning Task, user can take among two alternatives Semantic_Segmentation|Image_Classification')
parser.add_argument('--learning_model', dest='learning_model', type=str, default='CNN', help='Learning model used')
parser.add_argument('--criteria', dest = 'criteria', type = str, default = 'lithology', help = 'Characterization criteria of the substratum')
parser.add_argument('--pretrained_backbone', dest = 'pretrained_backbone', type=eval, choices=[True, False], default=False, help = 'Decide if the bockbone will be a pretrained one or will be trained from scratch')
parser.add_argument('--labels_type', dest='labels_type', type=str, default='onehot_labels', help='users can choose between onehot_labels(Image Classification) or multiple_labels(Multilabel Image Classification)')
# Hyperparameters
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number images in batch')
parser.add_argument('--compute_uncertainty', dest = 'compute_uncertainty', type = eval, choices = [True, False], default = True, help = 'Decide if will be computed the uncertainty measures')
parser.add_argument('--uncertainty_csv', dest = 'uncertainty_csv', type = eval, choices = [True, False], default = True, help = 'Decide if will be saved the uncertainty measures')
# Images pre-processing hyper-parameters
parser.add_argument('--resize', dest = 'resize', type = eval, choices = [True, False], default = True, help = 'Decide if resize will be applied')
parser.add_argument('--image_channels', dest='image_channels', type=int, default=3, help='Number of channels of images')
parser.add_argument('--new_size_rows', dest='new_size_rows', type=int, default=1024, help='Size of the random crop performed as Data Augmentation technique')
parser.add_argument('--new_size_cols', dest='new_size_cols', type=int, default=1024, help='Size of the random crop performed as Data Augmentation technique')
#Dirs
parser.add_argument('--dataset_main_path', dest='dataset_main_path', type=str, default='./static/resource/images/', help='Main path of the dataset images')
parser.add_argument('--checkpoints_main_path', dest='checkpoints_main_path', type=str, default='/home/zhang/Downloads/checkpoints', help='Path where checkpoints have been saved' )
parser.add_argument('--results_main_path', dest = 'results_main_path', default = './', help = 'Path where the results files will be saved')
args = parser.parse_args()

def main():
    args.checkpoints_main_path = args.checkpoints_main_path + '/trained_models/'
    print(args.checkpoints_main_path)

    args.results_main_path = args.results_main_path + 'results/'
    if not os.path.exists(args.results_main_path):
        os.makedirs(args.results_main_path)

    args.results_dir = args.results_main_path
    print("Dataset pre-processing...")
    dataset = Dataset(args)
    print("Checking the number of classifiers...")
    #checkpoint_files = os.listdir(args.checkpoints_main_path + args.criteria.lower() + '/')
    args.checkpoint_files = glob(args.checkpoints_main_path + args.criteria +  "/*/*/", recursive = True)
    args.number_classifier = len(args.checkpoint_files)
    print(args.checkpoint_files)
    print("Number of available classifiers: ", args.number_classifier)

    if len(args.checkpoint_files) > 0:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        args.save_results_dir = args.results_dir + args.criteria.lower() + '_' + dt_string + '/'
        if not os.path.exists(args.save_results_dir):
            os.makedirs(args.save_results_dir)
        print('[*]Initializing the substrate prediction system...')
        model = Model(args, dataset, True)
        print('[*]Models evaluation running...')
        model.Predict()
    else:
        print("The specified folder doesn't contain any valid checkpoint. Please check the folder path")

#if __name__ == '__main__':
#    main()
