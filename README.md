# Kaggle Diabetic Retinopathy Detection

### Installation
Install python dependencies via,
```
pip install -r requirements.txt
```
You need a CUDA capable GPU with at least 4GB of memory.  It is also recommended 
to have [CUDNN](https://developer.nvidia.com/cudnn) installed. The code should 
run without it but this hasn't been tested. The code has only been tested on 
python2 (not 3).

Extract train/test images to `data/train` and `data/test` respectively.

### Usage
#### Generating the kaggle solution
A commendet bash script to generate our final 2nd place solution can be found 
in `make_kaggle_solution.sh`.

Running all the commands sequentially will probaly take 7 - 10 days on recent
consumer grade hardware. If you have multiple GPUs you can speed things up
by doing training and feature extraction for the two networks at the same time. 
But due to the computationally heavy data augmentation it may be far less than
twice as fast especially when working with 512x512 pixel input images.

You can also obtain a quadratic weighted kappa score of 0.840 on the private
leaderboard by just training the 4x4 kernel networks and by performing only 50 
feature extraction iterations with the weights that gave you the best MSE 
validation scores. The entire ensemble achieves a score of 0.845.

#### Scripts
All these python scripts can be invoked with `--help` to display a brief help
message. They are meant to be executed in the order,

- `convert.py` crops and resizes images
- `train_nn.py` trains convolutional networks
- `transform.py` extracts features from trained convolutional networks
- `blend.py` blends features, optionally blending inputs from both patient eyes

##### convert.py
```
python convert.py --crop_size 128 --convert_directory data/train_tiny --extension tiff --directory data/train
python convert.py --crop_size 128 --convert_directory data/test_tiny --extension tiff --directory data/test
```
```
Usage: convert.py [OPTIONS]

Options:
  --directory TEXT          Directory with original images.  [default: data/train]
  --convert_directory TEXT  Where to save converted images.  [default: data/train_res]
  --test                    Convert images one by one and examine them on screen.  [default: False]
  --crop_size INTEGER       Size of converted images.  [default: 256]
  --extension TEXT          Filetype of converted images.  [default: tiff]
  --help                    Show this message and exit
```
##### train_nn.py
```
python train_nn.py --cnf configs/c_128_5x5_32.py
python train_nn.py --cnf configs/c_512_5x5_32.py --weights_from weigts/c_256_5x5_32/weights_final.pkl
```
```
Usage: train_nn.py [OPTIONS]

Options:
  --cnf TEXT           Path or name of configuration module.  [default: configs/c_128_4x4_tiny.py]
  --weights_from TEXT  Path to initial weights file.
  --help               Show this message and exit.
```

##### transform.py
```
python transform.py --cnf config/c_128_5x5_32.py --train --test --n_iter 5
python transform.py --cnf config/c_128_5x5_32.py --n_iter 5 --test_dir path/to/other/image/files
```
```
Usage: transform.py [OPTIONS]

Options:
  --cnf TEXT           Path or name of configuration module.  [default: config/c_128_4x4_32.py]
  --n_iter INTEGER     Iterations for test time averaging.  [default: 1]
  --skip INTEGER       Number of test time averaging iterations to skip. [default: 0]
  --test               Extract features for test set. Ignored if --train_dir is specified.  [default: False]
  --train              Extract features for test set. Ignored if --test_dir is specified.  [default: False]
  --weights_from TEXT  Path to weights file.
  --train_dir TEXT     Directory with training set images.
  --test_dir TEXT      Directory with test set images.
  --help               Show this message and exit.
```
##### blend.py
```bash
python blend.py --per_patient --tranform_file data/features/c_128_5x5_32_train_mean_iter_5_skip_0.npy
python blend.py --per_patient --directory data/features

```
```
Usage: blend.py [OPTIONS]

Options:
  --cnf TEXT            Path or name of configuration module.  [default: configs/c_128_4x4_32.py]
  --predict             Make predictions on test set features after training. [default: False]
  --per_patient         Blend features of both patient eyes.  [default: False]
  --features_file TEXT  Read features from specified file.
  --directory TEXT      Blend once for each (sub)directory and file in directory  [default: data/features]
  --n_iter INTEGER      Number of times to fit and average.  [default: 1]
  --help                Show this message and exit.
```

#### Configuration
Most of the convolutional network configuration is done via the files in the 
`config` directory.

