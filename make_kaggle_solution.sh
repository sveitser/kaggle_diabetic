#!/bin/bash

#
#       This script can be used to generate the 2nd place solution
#       of team o_O for the diabetic retinopath competition.
#

# terminate on error
set -e

########## Convert Images ##########

# Original images are expected to reside in data/{train,test}.
# Convert them to 512x512 pixel images.
python convert.py --crop_size 512 --convert_directory data/train_medium --extension tiff --directory data/train
python convert.py --crop_size 512 --convert_directory data/test_medium --extension tiff --directory data/test

# For the training set, make smaller images to train smaller versions of the 
# convolutional networks.
python convert.py --crop_size 256 --convert_directory data/train_small --extension tiff --directory data/train_medium
python convert.py --crop_size 128 --convert_directory data/train_tiny --extension tiff --directory data/train_medium

########## Train Convolutional Networks ##########

# Train network with 5x5 and 3x3 kernels.
python train_nn.py --cnf config/c_128_5x5_32.py
python train_nn.py --cnf config/c_256_5x5_32.py --weights_from weights/c_128_5x5_32/weights_final.pkl
python train_nn.py --cnf config/c_512_5x5_32.py --weights_from weights/c_256_5x5_32/weights_final.pkl

# Train network with 4x4 kernels.
python train_nn.py --cnf config/c_128_4x4_32.py
python train_nn.py --cnf config/c_256_4x4_32.py --weights_from weights/c_128_4x4_32/weights_final.pkl
python train_nn.py --cnf config/c_512_4x4_32.py --weights_from weights/c_256_4x4_32/weights_final.pkl

########## Extract Features ##########

# Please note that you can save a lot of time while maintaining most of the
# prediction accuracy by reducing the number of iterations from 50 down to 20 
# or even fewer.

# Extract features for network with 5x5 and 3x3 kernels.
BEST_VALID_WEIGHTS="$(ls -t weights/c_512_5x5_32/best/ | head -n 1)"
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --weights_from "$BEST_VALID_WEIGHTS"
# by default weights with best validation kappa druing train run are loaded
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --skip 50
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --skip 100 --weights_from weights/c_512_5x5_32/weights_final.pkl

# Extract features for network with 4x4 kernels.
BEST_VALID_WEIGHTS="$(ls -t weights/c_512_4x4_32/best/ | head -n 1)"
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --weights_from "$BEST_VALID_WEIGHTS"
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --skip 50
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --skip 100 --weights_from weights/c_512_4x4_32/weights_final.pkl

########## Blend Features ##########

# Link feature files for blending.
# For each feature extraction run we put the final mean and std deviation
# features into a separate directory. The blend.py script will then blend the 
# features for each feature extraction run separately and average the predictions
# before thresholding.

mkdir -p data/final_blend/{4x4,5x5}_{skip_0,skip_50,skip_100}
ln -s $PWD/data/features/c_512_4x4_32_train_{mean,std}_iter_50_skip_0.npy data/final_blend/4x4_skip_0
ln -s $PWD/data/features/c_512_4x4_32_train_{mean,std}_iter_50_skip_50.npy data/final_blend/4x4_skip_50
ln -s $PWD/data/features/c_512_4x4_32_train_{mean,std}_iter_50_skip_100.npy data/final_blend/4x4_skip_100
ln -s $PWD/data/features/c_512_5x5_32_train_{mean,std}_iter_50_skip_0.npy data/final_blend/5x5_skip_0
ln -s $PWD/data/features/c_512_5x5_32_train_{mean,std}_iter_50_skip_50.npy data/final_blend/5x5_skip_50
ln -s $PWD/data/features/c_512_5x5_32_train_{mean,std}_iter_50_skip_100.npy data/final_blend/5x5_skip_100

# validate
python blend.py --directory data/final_blend --per_patient

# make submission
python blend.py --directory data/final_blend --per_patient --predict

