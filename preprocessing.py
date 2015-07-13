""" The script transforms jpeg image files (from path_original_images). The brightness of every image is adjusted to the mean.
Moreover, the green channel is extracted, the gamma value is adjusted, the contrast is increased and the intensity level
is adjusted.
"""

import os
from skimage import data, feature, io, exposure, util, morphology
import numpy as np
from PIL import Image, ImageStat, ImageEnhance


class preprocessing():

    def __init__(self,path):
        self.names_all_images = os.listdir(path)
        self.input_path=path

    def PreprocessImages(self):
        # The mean brightness was already computed and it is 62.
        #mean_brightness=self.ComputeMeanBrightness(self)
        mean_brightness=62

        for element in self.names_all_images:
            # For the first 4lines, I used Pillow and after that I used skimage,
            # therefore I needed to save and load in between.
            current_image_loaded,currently_considered_image_path=self.ReadCurrentImage(element)
            image_equal_brightness=self.EqualiseBrightness(current_image_loaded, mean_brightness)
            extract_green_channel=self.ExtractGreenChannel(image_equal_brightness,currently_considered_image_path)
            self.SaveFinalImagePillow(extract_green_channel,currently_considered_image_path)

            current_image_loaded=self.LoadImageForSkimage(currently_considered_image_path)
            current_image_loaded=self.AdjustGamma(current_image_loaded)
            current_image_loaded=self.IncreaseContrast(current_image_loaded)
            current_image_loaded=self.EqualiseIntensity(current_image_loaded)
            self.SaveFinalImage(currently_considered_image_path,current_image_loaded)

    def ComputeMeanBrightness(self,names_all_images):
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

    def ReadCurrentImage(self,element):
        current_image_location=self.input_path+element
        location_to_save_image=os.getcwd() + '/data/results of preprocessing/'
        if not os.path.exists(location_to_save_image):
            os.makedirs(location_to_save_image)

        current_image_loaded=Image.open(current_image_location) # convert('L')
        currently_considered_image_path=location_to_save_image + element
        return current_image_loaded, currently_considered_image_path

    def EqualiseBrightness(self, current_image_loaded, mean_brightness):
        image_statisitcs_current_image=ImageStat.Stat(current_image_loaded)
        brightness_current_image=image_statisitcs_current_image.mean[0]
        percentage_brightness_enhancement=mean_brightness / brightness_current_image
        enhanceobj=ImageEnhance.Brightness(current_image_loaded)
        image_equal_brightness=enhanceobj.enhance(percentage_brightness_enhancement)
        return image_equal_brightness

    def ExtractGreenChannel(self,image_equal_brightness,currently_considered_image_path):
        extract_green_channel=np.array(image_equal_brightness)
        extract_green_channel[:,:,0]*=0 # red
        extract_green_channel[:,:,2]*=0 # blue
        extract_green_channel=Image.fromarray(extract_green_channel)
        return extract_green_channel

    def SaveFinalImagePillow(self,extract_green_channel,currently_considered_image_path):
        extract_green_channel.save(currently_considered_image_path)

    def LoadImageForSkimage(self,currently_considered_image_path):
        current_image_loaded=io.imread(currently_considered_image_path)
        return current_image_loaded


    def AdjustGamma(self,current_image_loaded):
        current_image_loaded=exposure.adjust_gamma(current_image_loaded,gamma=1,gain=1)
        return current_image_loaded

    def IncreaseContrast(self,current_image_loaded):
        current_image_loaded=exposure.equalize_adapthist(current_image_loaded)
        return current_image_loaded

    def EqualiseIntensity(self,current_image_loaded):
        current_image_loaded=exposure.rescale_intensity(current_image_loaded)
        return current_image_loaded

    def SaveFinalImage(self,currently_considered_image_path,current_image_loaded):
        io.imsave(currently_considered_image_path,current_image_loaded)

    def main(self):
        self.PreprocessImages()
        print('EOF')



# Instantiate class object and run it.
path_original_images = '/nas/kaggle/train/'
preprocessing_class=preprocessing(path_original_images)
preprocessing_class.main()
