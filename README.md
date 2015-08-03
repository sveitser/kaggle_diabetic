## Kaggle Diabetic Retinopathy Detection

### Installation
Install dependencies via,
```
pip install -r requirements.txt
```
Extract train/test images to ```data/train``` and ```data/test``` respectively.

### Usage
#### Generating the kaggle solution
A bash script to generate our final 2nd place solution can be found in 
`make_kaggle_solution.sh`.

Running all the commands sequentially will probaly take 7 - 10 days on good
consumer grade hardware. If you have multiple GPUs you can speed things up
by training the two networks at the same time. But due to the computationally
heavy data augmentation it may be far less than twice as fast especially when
working with the 512x512 pixel input images.

You can also obtain a quadratic weighted kappa score of 0.840 on the private
leaderboard by just training the 4x4 networks and by running only 50 feature 
extraction iterations with the weights that gave you the best validation scores. 
The entire ensemble achieves a score of 0.845.

#### Scripts
All these python scripts can be invoced with `-h` to display a brief help
message.

- `convert.py` crops and resizes images
- `train_nn.py` trains convolutional networks
- `transform.py` extracts features from trained convolutional networks
- `blend.py` blends features, optionally using inputs from both patient eyes

Example script usage
```bash
python convert.py --crop_size 128 --convert_directory data/train_tiny --extension tiff --directory data/train
python convert.py --crop_size 128 --convert_directory data/test_tiny --extension tiff --directory data/test
python train_nn.py --cnf config/c_128_5x5_32.py
python transform.py --cnf config/c_128_5x5_32.py --train --test --n_iter 5
python blend.py --per_patient --tranform_file data/transform/c_128_5x5_32_train_mean_iter_5_skip_0.npy
```

#### Configuration
Most of the convolutional network configuration is done via the files in the 
`config` directory.

