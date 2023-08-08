import zipfile
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
import numpy as np
import csv
import shutil

def unzip_file(zip_path, target_folder):
    # 检查目标文件夹是否存在，如果存在则删除
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取zip文件中的所有文件和文件夹列表
        file_list = zip_ref.namelist()
        file_list = [f for f in file_list if not f.startswith('__MACOSX/')]

        # 解压文件到目标路径
        for file in file_list:
            zip_ref.extract(file, target_folder)

    file_list.pop(0)
    # 移动文件到images文件夹下
    for file in file_list:
        source_path = os.path.join(target_folder, file)
        destination_path = os.path.join(target_folder, os.path.basename(file))
        shutil.move(source_path, destination_path)

    print('解压完成')


def image_label_generator(image_paths, labels, batch_size, datagen):
        num_samples = len(image_paths)
        while True:
            for offset in range(0, num_samples, batch_size):
                batch_image_paths = image_paths[offset:offset+batch_size]
                batch_labels = labels[offset:offset+batch_size]

                # 加载和增强图像
                batch_images = []
                for image_path in batch_image_paths:
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image = datagen.random_transform(image)
                    image = datagen.standardize(image)
                    batch_images.append(image)

                # 将图像和标签转换为数组
                batch_images = tf.stack(batch_images)
                batch_labels = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

                yield batch_images, batch_labels

def run():

    zip_file_path = './static/resource/images.zip'

    extract_dir = './static/images'

    unzip_file(zip_file_path, extract_dir)

    data_dir = './static/images'
    num_classes = 2

    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))

    labels = []
    for image in image_paths:
        label = os.path.basename(image).split('.')[0]
        labels.append(label)

    label_to_int = {label: i for i, label in enumerate(set(labels))}
    labels = [label_to_int[label] for label in labels]

    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_generator = image_label_generator(image_paths, labels, len(labels), datagen)

    # generate test data
    test_images, test_labels = next(test_generator)

    #define the model
    classifier = Sequential()
    classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    classifier.add(Conv2D(32,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    classifier.add(Flatten())
    classifier.add(Dense(units=128,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False)
    #classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    classifier = load_model('dogcat_model_bak.h5')

    # prediction
    predictions = classifier.predict(test_images)

    class_labels = ['cat', 'dog']

    predicted_labels = [class_labels[int(np.round(pred))] for pred in predictions]

    print(predicted_labels)
    # 设置CSV文件路径
    csv_file = 'predicted_labels.csv'

    # 检查文件是否存在
    if os.path.isfile(csv_file):
        # 如果文件存在，删除它
        os.remove(csv_file)

    # 将预测标签转换为包含每个标签的列表
    label_list = [label for label in predicted_labels]

    # Create a new CSV file and write the labels along with the image names
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'Predicted Label'])
        writer.writerows(zip(image_paths, predicted_labels))

    print('结果已存储为CSV文件')

    return predicted_labels
