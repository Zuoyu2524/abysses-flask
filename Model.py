import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import trange

from tensorflow.keras.backend import epsilon

from Networks import *
from Tools import *

class Model():
    def __init__(self, args, dataset, initializing):
        self.args = args
        self.dataset = dataset
        self.args.class_number = self.dataset.class_number
        self.initializing = initializing
        tf.set_random_seed(int(time.time()))
        tf.reset_default_graph()
        self.data = tf.placeholder(tf.float32, [None, self.args.new_size_rows, self.args.new_size_cols, self.args.image_channels], name = "data")
        if self.initializing:
            self.args.backbone_name = self.args.checkpoint_files[0].split('/')[-3]
            self.args.trained_model_path = self.args.checkpoint_files[0]

        self.model = Networks(self.args)
        Classifier_Outputs = self.model.learningmodel.build_Model(self.data, reuse = False, name = "CNN" )
        self.prediction_c = Classifier_Outputs[-1]

        self.saver = tf.train.Saver(max_to_keep=5)
        self.sess=tf.Session()
        self.sess.run(tf.initialize_all_variables())

        mod = self.load(self.args.trained_model_path)
        if not self.initializing:
            if mod:
                print(" [*] Load with SUCCESS")
            else:
                print(" [!] Load failed...")
                sys.exit()

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return aux
        else:
            return ''

    def save_to_csv(self, dataframe, output_file, columns):
        df_selected = dataframe[columns]
        df_selected.columns = ['Image Name', 'MajorityVoting', 'Predicted Label']  
        df_selected.to_csv(output_file, index=False)


    def Predict(self):
        args = self.args
        dataset = self.dataset
        #Computing the number of batches
        num_batches_ts = len(self.dataset.file_paths)//self.args.batch_size
        batchs = trange(num_batches_ts)

        if self.args.labels_type == 'multiple_labels':
            Predicted_Labels = np.zeros((num_batches_ts * self.args.batch_size, self.args.class_number))

        if self.args.compute_uncertainty:
            predictive_variance = []
            predictive_entropy  = []
            mutual_information  = []

        if self.args.uncertainty_csv:
            File_Names = []
            Predicted_Categories = []
            Predicted_Labels = []
            MajorityVoting = []

        if self.args.labels_type == 'multiple_labels':
            Predicted_Labels = np.zeros((num_batches_ts * self.args.batch_size, self.args.class_number))

        for b in batchs:
            paths_batch = self.dataset.file_paths[b * self.args.batch_size : (b + 1) * self.args.batch_size]

            file_name = paths_batch[0].split('/')[-1][:-4]
            if self.args.compute_uncertainty and self.args.uncertainty_csv:
                File_Names.append(file_name)

            data_batch = self.dataset.read_samples(paths_batch)
            backbone_i = 0
            voting_array = np.zeros((1 , self.dataset.class_number))
            likelihood_array = np.zeros((self.args.number_classifier,self.dataset.class_number))
            for checkpoint in self.args.checkpoint_files:
                backbone = checkpoint.split('/')[-3]
                self.args.backbone_name = backbone
                self.args.trained_model_path = checkpoint
                self.__init__(args, dataset, False)
                batch_prediction_ = self.sess.run(self.prediction_c, feed_dict={self.data: data_batch})
                likelihood_array[backbone_i, :] = batch_prediction_

                if self.args.labels_type == 'onehot_labels':
                    voting_array[0, np.argmax(batch_prediction_, axis = 1)[0]] += 1
                if self.args.labels_type == 'multiple_labels':
                    y_pred = ((batch_prediction_ > 0.5) * 1.0)
                    voting_array += y_pred

                backbone_i += 1

            #print(likelihood_array)
            batch_prediction = voting_array.copy()
            #print(batch_prediction)
            MajorityVoting.append(batch_prediction[0])

            if self.args.labels_type == 'onehot_labels':
                y_pred = np.argmax(batch_prediction, axis = 1)
                Predicted_Labels.append(y_pred[0])

            if self.args.labels_type == 'multiple_labels':
                y_pred = ((batch_prediction > 15) * 1.0)
                Predicted_Labels[b * self.args.batch_size : (b + 1) * self.args.batch_size, :] = y_pred[0,:]

            if self.args.labels_type == 'onehot_labels':
                Predicted_Categories.append(self.dataset.class_names[y_pred[0]])
            if self.args.labels_type == 'multiple_labels':
                true_names = '_'
                predicted_names = '_'
                for i in range(len(y_pred[0,:])):
                    if y_pred[0,i] == 1:
                        predicted_names = predicted_names + self.dataset.class_names[i] + '_'

                Predicted_Categories.append(predicted_names)

            if self.args.compute_uncertainty:
            # Predictive Variance
                predictive_variance_k = np.var(likelihood_array, axis = 0)
                predictive_variance.append(np.mean(predictive_variance_k))
            # Predictive Entropy
                mean_likelihood_k = np.mean(likelihood_array, axis = 0)
                predictive_entropy.append(-1 * np.mean(mean_likelihood_k * np.log(mean_likelihood_k + 1e-5)))
            # Mutual Information
                classifiers_entropy = np.zeros((self.args.number_classifier, 1))
                for b in range(self.args.number_classifier):
                    classifiers_entropy[b, 0] = -1 * np.mean(likelihood_array[b, :] * np.log(likelihood_array[b, :] + 1e-5))

                mutual_information.append(-1 * np.mean(mean_likelihood_k * np.log(mean_likelihood_k)) - np.mean(classifiers_entropy))

        if self.args.uncertainty_csv:
            print(len(File_Names), len(MajorityVoting), len(Predicted_Labels), len(Predicted_Categories), len(predictive_variance), len(predictive_entropy), len(mutual_information))
            df = pd.DataFrame({'File_Names': File_Names,
                               'MajorityVoting': MajorityVoting,
                               #'Prediction': Predicted_Labels,
                               'Predicted_Categorie': Predicted_Categories,
                               'PV': predictive_variance,
                               'PE': predictive_entropy,
                               'MI': mutual_information})
            df.to_csv(self.args.save_results_dir + r'Results_file.csv', index=False)
            output_file = "predicted_labels.csv"
            columns = ['File_Names', 'MajorityVoting', 'Predicted_Categorie']
            self.save_to_csv(df, output_file, columns)
            print(df)
