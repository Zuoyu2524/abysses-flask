import os
import sys
import argparse

from Tools import *
parser = argparse.ArgumentParser(description='')

parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_Interactive', help='Decide wether the script will be running')
args = parser.parse_args()

Schedule = []

CRITERIA_NAME  = ['lithology','SW_fragments','morphology']
DATASET_MAIN_PATH = '/datawork/DATA/SAPIN_Original/'
if args.running_in == 'Datarmor_Interactive':
    MAIN_COMMAND = "Main.py"
if args.running_in == 'Datarmor_PBS':
    MAIN_COMMAND = "$HOME/CODE/SubstratuM/Main.py"


for criteria in CRITERIA_NAME:

    if criteria == 'morphology':
        labels_type = 'multiple_labels'
    else:
        labels_type = 'onehot_labels'


    Schedule.append("python " + MAIN_COMMAND + " --train_task Image_Classification --learning_model CNN --pretrained_backbone False --criteria " + criteria + " --labels_type " + labels_type + " "
                    "--compute_uncertainty True --uncertainty_csv True --batch_size 1 "
                    "--resize True --image_channels 3 --new_size_rows 1024 --new_size_cols 1024 "
                    "--dataset_main_path " + DATASET_MAIN_PATH + " "
                    "--checkpoints_main_path /datawork/EXPERIMENTS/LOIC/ "
                    "--results_main_path /datawork/EXPERIMENTS/LOIC/")

for i in range(len(Schedule)):
    os.system(Schedule[i])
