__author__ = 'stephan'

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
from pandas.io.parsers import read_csv
from pandas import Series, DataFrame
from skimage import io
from skimage.transform import resize
from sklearn import cross_validation
import numpy as np
from numpy import array
import os
import sys
import cv2
from PIL import Image
import util


class cnn():

    def create_cnn(self, image_parameters):
        input_shape_calculated = image_parameters[0] * image_parameters[1]
        net1 = NeuralNet(
            layers=[
                ('input',layers.InputLayer),
                ('hidden', layers.DenseLayer),
                ('output',layers.DenseLayer),
                ],
            #Parameters
            input_shape=(None,input_shape_calculated),
            hidden_num_units=100,
            output_nonlinearity=None,
            output_num_units=2,

            #Optimisation method:
            update=nesterov_momentum,
            update_learning=0.01,
            update_momentum=0.9,

            regression=True,
            max_epochs=400,
            verbose=1,
        )
        return net1



    def main(self):
        image_directory = '/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/train_original/'
        input_only_left = True
        load_the_data = load_data(input_only_left,image_directory)
        labels_images, image_size, loaded_images = load_the_data.load_labels_and_image_size()



        net = self.create_cnn(image_size)
        net.fit(loaded_images,labels_images)


        X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(loaded_images, labels_images, test_size = 0.2, random_state = 0)

        # problem: 25000 loaded images (should be 50) and 50 labels



        load_data_object = load_data()
        load_data_object.load_labels_and_image_size(load_data_object)


        net1.fit(self,x,y)


        
        # do something



class load_data():
    def __init__(self,input_only_left, path_original_images_on_hdd):
        self.only_left=input_only_left
        self.path_original_images = path_original_images_on_hdd

    def prepare_data(self,path_labels):
        labels_images = self.read_in_labels(path_labels)
        self.ensure_labeled_images_exist(labels_images)
        labels_images = self.delete_erroneous_entries(labels_images)
        image_size = self.obtain_image_size()

        # shuffle
        labels_images = labels_images.reindex(np.random.permutation(labels_images.index))
        labels_images = labels_images[0:50]
        #labels_images = self.make_labels_binary(labels_images)
        return labels_images, image_size

    def delete_erroneous_entries(self, labels_images):
        list_existing_images = os.listdir(self.path_original_images)

        logical_list = []
        for element in labels_images.iloc[:,0]:
            if not(str(element+'.jpeg') in list_existing_images):
                logical_list.append(False)
            else:
                logical_list.append(True)

        labels_images = labels_images[logical_list]
        return labels_images



    def load_images(self,labels_images):
        counter = 0
        for row in labels_images.iloc[:,0]:
            if not('image' in row): # skip first line
                filename = self.path_original_images + row + '.jpeg'
                current_image = cv2.imread(filename)
                current_image = cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
                current_image = cv2.resize(current_image,(500,500))
                current_image = current_image.reshape(1,500,500)

                if counter == 0:
                    collection_images = current_image
                else:
                    collection_images = np.vstack((collection_images,current_image))
            counter= counter + 1
        print collection_images.shape
        return collection_images

    def read_in_labels(self,path_labels):
        labels_images = read_csv(path_labels)
        if self.only_left:
            counter = 0
            logical_list = []
            for element in labels_images.iloc[:,0]:
                if not('left' in element):
                    logical_list.append(True)
                else:
                    logical_list.append(False)
        labels_images = labels_images[logical_list]
        return labels_images

    def ensure_labeled_images_exist(self,labels_images):
        if labels_images.size == len(os.listdir(self.path_original_images)):
            print("The number of labels and the number of images are equivalent.")
        else:
            print("The number of labels and the number of images are NOT equivalent!!")
            print "Number of images: ", len(os.listdir(self.path_original_images))
            print "Number of labels: ", labels_images.size

    def make_labels_binary(self,labels_images):
        counter = 0
        for element in labels_images.iloc[:,1]:
            if element == 2 or element == 3 or element == 4:
                labels_images.iloc[counter,1] = 1
            counter = counter + 1
        print("Labels have been succesfully binarized!")

    def obtain_image_size(self):
        all_images = os.listdir(self.path_original_images)
        path_of_single_image = self.path_original_images + '/' + all_images[0]
        current_image_opened = Image.open(path_of_single_image)
        width, height = current_image_opened.size
        image_parameters = Series([width, height], ['width', 'height'])
        return image_parameters

    def load_labels_and_image_size(self):
        path_labels = os.getcwd() + '/data/trainLabels.csv'
        labels_images, image_size = self.prepare_data(path_labels)
        loaded_images = self.load_images(labels_images)
        return labels_images,image_size, loaded_images
        #load_the_data.load_images(labels_images)




#image_directory = '/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/train_original/'
#input_only_left = True
#load_the_data = load_data(input_only_left,image_directory)
#labels_images, image_size = load_the_data.load_labels_and_image_size()

# Idea Mathis
# 1. Shuffle data
# 2. Mention noise


new_cnn = cnn()
new_cnn.main()






