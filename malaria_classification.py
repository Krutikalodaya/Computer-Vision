import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

class malaria_classification():
    def __init__(self, image):
        self.image = image
    def labeling_data(self):
        '''
        This function is intended to label the images into parasited images and uninfected images.
        Please make sure only .png files are included in each of these folders.
        Parasited images are labeled 0 and Uninfected images are labeled 1.
        '''
        SIZE = 64
        dataset = []
        label = []
        parasitized_images = os.listdir(self.image + 'Parasitized/')
        for i, image_name in enumerate(parasitized_images):
            try:
                if (image_name.split('.')[1] == 'png'):
                    image = cv2.imread(self.image + 'Parasitized/' + image_name)
                    image = Image.fromarray(image, 'RGB')
                    image = image.resize((SIZE, SIZE))
                    dataset.append(np.array(image))
                    label.append(0)
            except Exception:
                print("Could not read image {} with name {}".format(i, image_name))

        uninfected_images = os.listdir(self.image + 'Uninfected/')
        for i, image_name in enumerate(uninfected_images):
            try:
                if (image_name.split('.')[1] == 'png'):
                    image = cv2.imread(self.image + 'Uninfected/' + image_name)
                    image = Image.fromarray(image, 'RGB')
                    image = image.resize((SIZE, SIZE))
                    dataset.append(np.array(image))
                    label.append(1)
            except Exception:
                print("Could not read image {} with name {}".format(i, image_name))
        image_data = np.array(dataset)
        labels = np.array(label)
        idx = np.arange(image_data.shape[0])
        #Random shuffle is done here to have minimum value of a loss function
        np.random.shuffle(idx)
        self.image_data = image_data[idx]
        self.labels = labels[idx]
        return parasitized_images,uninfected_images

    def training_set(self):
        #Dataset is divieded into training and testing set.
        x_train, x_test, y_train, y_test = train_test_split(self.image_data, self.labels, test_size=0.2, random_state=101)
        y_train = np_utils.to_categorical(y_train, num_classes=2)
        y_test = np_utils.to_categorical(y_test, num_classes=2)
        return x_train,y_train,x_test,y_test

    def data_visualization(self):
        #Visualize the images of both parasited and uninfected images using matplotlib.pyplot
        parasitized_images,uninfected_images = self.labeling_data()
        print(parasitized_images)
        plt.figure(figsize=(12, 12))
        #4 images are shown for parasited image
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            img = cv2.imread(self.image + 'Parasitized/' + parasitized_images[i])
            plt.imshow(img)
            plt.title('PARASITIZED : 1')
            plt.tight_layout()
        plt.show()
        #4 images are shown for uninfected image
        plt.figure(figsize=(12, 12))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            img = cv2.imread(self.image + 'Uninfected/' + uninfected_images[i])
            plt.imshow(img)
            plt.title('UNINFECTED : 1')
            plt.tight_layout()
        plt.show()

    def CNNbuild(self, height, width, classes, channels):
        '''
        Building Sequential CNN model
        :param height: pixel hieght of the image
        :param width: pixel width of the image
        :param classes: number of classees (classification classes)
        :param channels: depth/RGB channels of the image
        :return: model
        '''
        model = Sequential()

        inputShape = (height, width, channels)
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (channels, height, width)
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=inputShape))
        model.add(MaxPooling2D(2, 2))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

    def main(self):
        '''
        Data visualization is called here
        CNN model is created and compiled with categorical cross entropy (please go through it if you don't know what that is)
        optimization is done using adam optimization algorithm.
        :return: prediction, loss and accuracy
        '''
        self.data_visualization()
        height = 64
        width = 64
        classes = 2
        channels = 3
        model = self.CNNbuild(height=height, width=width, classes=classes, channels=channels)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        x_train,y_train,x_test,y_test = self.training_set()
        h = model.fit(x_train, y_train, epochs=20, batch_size=32)
        plt.figure(figsize=(18, 8))
        plt.plot(range(20), h.history['acc'], label='Training Accuracy')
        plt.plot(range(20), h.history['loss'], label='Taining Loss')
        # ax1.set_xticks(np.arange(0, 31, 5))
        plt.xlabel("Number of Epoch's")
        plt.ylabel('Accuracy/Loss Value')
        plt.title('Training Accuracy and Training Loss')
        plt.legend(loc="best")
        predictions = model.evaluate(x_test, y_test)
        print(f'LOSS : {predictions[0]}')
        print(f'ACCURACY : {predictions[1]}')

obj = malaria_classification("yourpath\cell-images-for-detecting-malaria\cell_images\\")
obj.main()
