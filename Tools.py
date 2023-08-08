import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def preprocess_input(data, backbone_name):
    if backbone_name is None:
        data = data/255.
    elif backbone_name == 'movilenet':
        data = tf.keras.applications.mobilenet.preprocess_input(data)
    elif backbone_name == 'resnet50':
        data = tf.keras.applications.resnet.preprocess_input(data)
    elif backbone_name == 'vgg16':
        data = tf.keras.applications.vgg16.preprocess_input(data)

    return data

def create_dict(image, label):
    return {"image": image, "label": label}

def compute_metrics(y_true, y_pred, average):
    accuracy = 100*accuracy_score(y_true, y_pred)
    f1score = 100*f1_score(y_true, y_pred, average=average)
    recall = 100*recall_score(y_true, y_pred,average=average)
    precision = 100*precision_score(y_true, y_pred,average=average)
    return accuracy, f1score, precision, recall

def plottsne_features(features, labels, save_path, epoch ,USE_LABELS = True):
    save_path = save_path + '/scatter_plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    colors = []
    plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    if USE_LABELS:
        colors.append('#1724BD')
        colors.append('#0EB7C2')
        colors.append('#BF114B')
        colors.append('#E98E2C')
        colors.append('#008f39')
        colors.append('#663300')
        colors.append('#8D1354')
        for i in range(features.shape[0]):
            # plot colored number
            ax.plot(features[i, 0], features[i, 1], marker='o',
                      color=colors[int(labels[i,0])])
    else:
        for i in range(features.shape[0]):
            # plot colored number
            ax.plot(features[i, 0], features[i, 1], marker='o',
                      color='b')

    plt.savefig(save_path + str(epoch) + '_scatter_plot.png')
    plt.clf()

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, heatmap, thresh, save_path, emphasize=False):

    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.


    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    #heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)

    heatmap = np.uint8(255 * heatmap)
    plt.figure(figsize=(20,20))
    ax = plt.subplot(111)
    plt.imshow(img_bgr)
    plt.imshow(heatmap, 'jet', interpolation='none', alpha=0.30)
    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    #hif = .8
    #superimposed_img = heatmap * hif + img_bgr
    #superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255
    #superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    plt.savefig(save_path)
    plt.clf()
    #return superimposed_img_rgb

def Recover_hyperparameters_GM(file_path, b, c):

    continue_ = False

    if os.path.exists(file_path):
        t = open(file_path, "r")
        lines = t.readlines()
        l = lines[-1]
        if "Completed" not in l:
            continue_ = True
            fields = l.split('/')
            b = int(fields[0])
            c = int(fields[1])
        t.close()
    return continue_, b, c

def Recover_hyperparameters_MS(args):
    continue_ = False

    if os.path.exists(args.tracking_files + "Model_tracking_" + args.identifier + ".txt"):

        t = open(args.tracking_files + "Model_tracking_" + args.identifier + ".txt")
        lines = t.readlines()
        l = lines[-1]
        fields = l.split('/_')
        if args.csvfile_name in fields[0]:
            continue_ = True
            args.save_checkpoint_path = fields[0]
            args.r = int(fields[1])
            args.initial_epoch = int(fields[2]) + 1
            args.best_val_fs = float(fields[3])
    return continue_, args
