import csv
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing import image as keras_image
import utils

def get_data(path):
    '''
        Read csv from file.

        - Get center, left and right images.
        - Associate steering angle with each image.
        - Apply angle correction on left and right images.
    '''

    images, angles = [], []
    CORRECTION = .2
    with open(path) as csvfile:
        for center, left, right, steer, _, _, _ in csv.reader(csvfile):
            steer = float(steer)
            images.extend([center, left, right])
            angles.extend([steer, steer+CORRECTION, steer-CORRECTION])
    return shuffle(np.array(images), np.array(angles))

def generator(X_train, y_train, batch_size, training=False):
    '''
        Data generator used for training the model.

        - Reades images from path
        - Preprocess images
        - Augments training data only.
    '''

    number_samples = len(y_train)
    while 1:
        shuffle(X_train, y_train)
        for offset in range(0, number_samples, batch_size):
            X, y = X_train[offset:offset+batch_size], y_train[offset:offset+batch_size]
            X = [utils.open_image(x) for x in X]
            X = [utils.pre_process(x) for x in X]

            if training:
                for i, (image, label) in enumerate(zip(X, y)):
                    X[i] = utils.augment(image)

            X = np.array([keras_image.img_to_array(x) for x in X])
            yield shuffle(X, y)

