#!/bin/bash

# Original images are expected to reside in data/{train,test}.
# Convert them to 512x512 pixel images.
python convert.py --crop_size 512 --convert_directory data/train_medium --extension tiff --directory data/train
python convert.py --crop_size 512 --convert_directory data/test_medium --extension tiff --directory data/test

# For the training set, make smaller images to train smaller versions
# of the final networks.
python convert.py --crop_size 256 --convert_directory data/train_small --extension tiff --directory data/train_medium
python convert.py --crop_size 128 --convert_directory data/train_tiny --extension tiff --directory data/train_medium

# train network with 5x5 and 3x3 kernels
python train_nn.py --cnf config/c_128_5x5_32.py # 2 hours
python train_nn.py --cnf config/c_256_5x5_32.py --weights_from weights/c_128_5x5_32/weights_final.pkl # 8 hours
python train_nn.py --cnf config/c_512_5x5_32.py --weights_from weights/c_256_5x5_32/weights_final.pkl # 2 days

# train network with 4x4 kernels
python train_nn.py --cnf config/c_128_4x4_32.py # 2 hours
python train_nn.py --cnf config/c_256_4x4_32.py --weights_from weights/c_128_4x4_32/weights_final.pkl # 8 hours
python train_nn.py --cnf config/c_512_4x4_32.py --weights_from weights/c_256_4x4_32/weights_final.pkl # 2 days

# Extract features (takes about 1 day for each set of 50 iterations).
# Please not that you can save a lot of time while maintaining most of the
# prediction accuracy by reducing the number of iterations.
BEST_VALID_WEIGHTS="$(ls -t weights/c_512_5x5_32/best/ | head -n 1)"
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --weights_from "$BEST_VALID_WEIGHTS"
# by default weights with best validation kappa are loaded
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --skip 50
python transform.py --cnf config/c_512_5x5_32.py --train --test --n_iter 50 --skip 100 --weights_from weights/c_512_5x5_32/weights_final.pkl

BEST_VALID_WEIGHTS="$(ls -t weights/c_512_4x4_32/best/ | head -n 1)"
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --weights_from "$BEST_VALID_WEIGHTS"
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --skip 50
python transform.py --cnf config/c_512_4x4_32.py --train --test --n_iter 50 --skip 100 --weights_from weights/c_512_4x4_32/weights_final.pkl

# link feature files for blending
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

