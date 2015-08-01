# Standard Library
import os
import csv

# Third party library
import shutil
from skimage import data, feature, io, exposure, util, morphology
import cv2
import SimpleCV


import numpy as np
from PIL import Image, ImageStat, ImageEnhance


class extractfeatures():

    def __init__(self):
        self.names_all_images = os.listdir(os.getcwd() + '/data/preprocessing/')
        #self.input_path=  os.getcwd() + '/data/Sample/'
        self.input_path='/media/stephan/12C3-91C8/Kaggle.com/diabetic_retinopathy/kaggle diabetic_/train_original/'


    def SelectAndCopyRelevantImages(self,some_argument):

        # Auxilliary functions
        def ExtractRelevantFilenames(self,path_of_csv_file):
            list_of_level_1_diagnosis = []
            max_iterations=50
            current_iteration =1

            with open (path_of_csv_file,'rb') as training_labels:
                content_of_training_labels = csv.reader(training_labels)
                for row in content_of_training_labels:
                    if current_iteration <= max_iterations:
                        file_path = self.input_path + row[0]+'.jpeg' #'.tiff'

                        if row[1]=='1' and os.path.isfile(file_path):
                            list_of_level_1_diagnosis.append(row[0])
                            current_iteration=current_iteration + 1

            return list_of_level_1_diagnosis

        def CopyRelevantFilesToNewLocation(self,list_of_level_1_diagnosis):
            for element in list_of_level_1_diagnosis:
                # For the input , the original images from kaggle.com are used (and not the resized ones)
                old_path = self.input_path + element + '.jpeg'
                new_path = os.getcwd() + '/data/preprocessing/'  + element + '.jpeg'
                shdata.copyfile(old_path, new_path)
            self.names_all_images=os.listdir(os.getcwd() + '/data/preprocessing/')


        def main(self):
            path_of_csv_file=os.getcwd()+'/data/trainLabels.csv'
            list_of_level_1_diagnosis=ExtractRelevantFilenames(self,path_of_csv_file)
            CopyRelevantFilesToNewLocation(self,list_of_level_1_diagnosis)

        # Run it
        main(self)


    def read_out_brightness_all_pictures(self,names_all_images):
        single_brightness_values = np.empty(len(self.names_all_images))
        counter = 0
        for element in self.names_all_images:
            current_image_location=self.input_path+element
            current_image_loaded=Image.open(current_image_location).convert('L')
            brightness_current_image=ImageStat.Stat(current_image_loaded)
            single_brightness_values[counter]=brightness_current_image.mean[0]
            counter=counter+1

        mean_brightness= round(single_brightness_values.mean()).as_integer_ratio()[0]
        print(mean_brightness)
        return mean_brightness


    def PreprocessImages(self):

        # Auxilliary functions
        def ReadCurrentImage(self,element):
            current_image_location=self.input_path+element
            location_to_save_image=os.getcwd() + '/data/results of preprocessing/'
            current_image_loaded=Image.open(current_image_location) # convert('L')
            currently_considered_image_path=location_to_save_image + element
            return current_image_loaded, currently_considered_image_path

        def PILToEqualiseBrightnessExtractGreenChannel(self, current_image_loaded, mean_brightness,currently_considered_image_path):
            image_statisitcs_current_image=ImageStat.Stat(current_image_loaded)
            brightness_current_image=image_statisitcs_current_image.mean[0]
            percentage_brightness_enhancement=mean_brightness / brightness_current_image
            enhanceobj=ImageEnhance.Brightness(current_image_loaded)
            enhanced_image=enhanceobj.enhance(percentage_brightness_enhancement)

            extract_green_channel=np.array(enhanced_image)
            extract_green_channel[:,:,0]*=0 # red
            extract_green_channel[:,:,2]*=0 # blue
            extract_green_channel=Image.fromarray(extract_green_channel)

            extract_green_channel.save(currently_considered_image_path)


        def AdjustGammaIncreaseContrastEqualiseIntensity(currently_considered_image_path):
            current_image_loaded=io.imread(currently_considered_image_path)
            current_image_loaded=exposure.adjust_gamma(current_image_loaded,gamma=1,gain=1)
            current_image_loaded=exposure.equalize_adapthist(current_image_loaded)
            current_image_loaded=exposure.rescale_intensity(current_image_loaded)
            io.imsave(currently_considered_image_path,current_image_loaded)


        def main(self):
            mean_brightness=self.read_out_brightness_all_pictures(self)

            for element in self.names_all_images:

                current_image_loaded,currently_considered_image_path=ReadCurrentImage(self,element)

                PILToEqualiseBrightnessExtractGreenChannel(self, current_image_loaded, mean_brightness,currently_considered_image_path)

                AdjustGammaIncreaseContrastEqualiseIntensity(currently_considered_image_path)


        main(self)


    def BlobDetection(self,something):
        for element in self.names_all_images:
            location_of_image=os.getcwd()+'/data/results of preprocessing/'+element

            print(os.path.exists(location_of_image))

            current_image = SimpleCV.Image(location_of_image).grayscale()

            #current_image_adjusted = current_image.binarize() # binarise is a very drastic approach

            #current_image_adjusted.save(os.getcwd() + '/test.jpeg')


            current_image_adjusted = current_image.invert()

            #current_image_adjusted.save(os.getcwd() + 'inverted.jpeg')

            blobs_of_image = current_image_adjusted.findBlobs()

            print "Number of Blobs ", blobs_of_image.count()
            print "Areas: ", blobs_of_image.area()








    def __call__(self):
        CopyAndLowLevelPreprocessImages = 0

        if CopyAndLowLevelPreprocessImages == 1:
            self.SelectAndCopyRelevantImages(self)
            self.PreprocessImages(self)

        self.BlobDetection(self)
        print('EOF')



# Instatiate class object and run it
run_class=extractfeatures()
run_class.__call__()





## Random code for blob detection

## Second approach for blob detection
# """
#from matplotlib import pyplot as plt
#from skimage import data
#from skimage.feature import blob_dog, blob_log, blob_doh
#from math import sqrt
#from skimage.color import rgb2gray

#image = data.hubble_deep_field()[0:500, 0:500]
#image_gray = rgb2gray(image)

# blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
# Compute radii in the 3rd column.
#blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

#blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

#blobs_list = [blobs_log, blobs_dog, blobs_doh]
#colors = ['yellow', 'lime', 'red']
#titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
#          'Determinant of Hessian']
#sequence = zip(blobs_list, colors, titles)

#for blobs, color, title in sequence:
#   fig, ax = plt.subplots(1, 1)
#  ax.set_title(title)
#  ax.imshow(image, interpolation='nearest')
#  for blob in blobs:
#        y, x, r = blob
#      c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
#      ax.add_patch(c)

#plt.show()"""





            #image=cv2.imread(location_of_image, cv2.IMREAD_GRAYSCALE)

            # Set up the detector with default parameters.
            #detector=cv2.SimpleBlobDetector()

            # Setup SimpleBlobDetector parameters.
            #params = cv2.SimpleBlobDetector_Params()

            # Change thresholds
            #params.minThreshold = 1;
            #params.maxThreshold = 1000000;

            # Filter by Area.
            #params.filterByArea = True
            #params.minArea = 100

            # Filter by Circularity
            #params.filterByCircularity = True
            #params.minCircularity = 0.1

            # Filter by Convexity
            #params.filterByConvexity = True
            #params.minConvexity = 0.87

            # Filter by Inertia
            #params.filterByInertia = True
            #params.minInertiaRatio = 0.01

            # Create a detector with the parameters
            #ver = (cv2.__version__).split('.')
            #if int(ver[0]) < 3 :
            #    detector = cv2.SimpleBlobDetector(params)
            #else :
            #    detector = cv2.SimpleBlobDetector_create(params)

            # Detect blobs.
            #keypoints=detector.detect(image)

            #print(keypoints)

            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            #im_with_keypoints=cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


            # Show keypoints
            #cv2.imshow("Keypoints",im_with_keypoints)
            #cv2.waitKey(0)

            #print "largest green blob at " + str(green_blobs[0].x) + ", " + str( green_blobs[0].y)


# further instruction for preprocessing

                # Binarize and Invert

                # Perform White Top-Hat transformation (see below)

                # http://scikit-image.org/docs/dev/auto_examples/plot_blob.html for blob detection
                #feature.blob_dog

                #http://scikit-image.org/docs/dev/api/skimage.feature.html

                # perform fovea extraction
                # perform blood vessel extraction
                # maybe convolution (see Patwari paper)
                # perform blob detection

                #preprocessed_image_before_tophat = img_as_ubyte(io.imread(currently_considered_image_path, as_grey=True))
                #selem = disk(8)

                #image_after_tophat_transform = white_tophat(preprocessed_image_before_tophat, selem)
                #io.imsave(currently_considered_image_path,image_after_tophat_transform)