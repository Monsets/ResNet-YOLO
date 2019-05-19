import keras
import numpy as np
import cv2

from utils import area_in_rect


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, conf, file_list, n_channels=3):
        'Initialization'
        self.dim = (conf.batch_size, conf.resize_shape, conf.resize_shape)
        self.batch_size = conf.batch_size
        self.conf = conf
        self.file_list = file_list
        self.n_channels = n_channels
        self.n_classes = conf.num_classes
        self.shuffle = conf.shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, f):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        images = []
        y_true = []
        for line in f:
            # read image
            image = cv2.imread(line.split(' ')[0])
            image = cv2.resize(image, (self.conf.resize_shape, self.conf.resize_shape))
            images.append(image)
            # get shape
            orig_shape_x = image.shape[1]
            orig_shape_y = image.shape[0]
            # resizing coef
            x_change = self.conf.resize_shape / orig_shape_x
            y_change = self.conf.resize_shape / orig_shape_y

            # bounding boxes
            labels = line.split(' ')[1:]
            # split coordinates of bounding boxes
            for i in range(len(labels)):
                labels[i] = labels[i].strip().split(',')
            # convert to int
            labels = np.array(labels).astype(int)
            # resize according to new image sizes
            for label in range(len(labels)):
                labels[label][0] *= x_change
                labels[label][1] *= y_change
                labels[label][2] *= x_change
                labels[label][3] *= y_change

            true_labels = np.zeros((self.conf.num_areas, self.conf.num_areas, self.conf.B * 5 + self.conf.num_classes))

            coords = [i for i in range(0, self.conf.resize_shape, self.conf.resize_shape // self.conf.num_areas)]
            coords.append(self.conf.resize_shape)
            # for every box
            for x in range(self.conf.num_areas):
                for y in range(self.conf.num_areas):
                    for rect in labels:
                        # if area lies within bounding box add it to label
                        if area_in_rect((coords[x], coords[y], coords[x + 1] - coords[x], coords[y + 1] - coords[y]),
                                        rect[:4]) > self.conf.area_threshold:
                            true_labels[x][y][0] = rect[0]
                            true_labels[x][y][1] = rect[1]
                            true_labels[x][y][2] = rect[2]
                            true_labels[x][y][3] = rect[3]
                            true_labels[x][y][4] = 1
                            true_labels[x][y][rect[4] + 1] = 1

                            break
            y_true.append(true_labels)

        return np.array(images), np.array(y_true)
