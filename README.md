# Kaggle Diabetic Retinopathy Detection

### Installation

Extract train/test images to `data/train` and `data/test` respectively.

Install python dependencies via,
```
pip install -r requirements.txt
```
You need a CUDA capable GPU with at least 4GB of memory.  It is also recommended 
to have [CUDNN](https://developer.nvidia.com/cudnn) installed. The code should 
run without it but this hasn't been tested. The code has only been tested on 
python2 (not 3).

Code was developed and tested on arch linux and hardware with a i7-2600k CPU,
GTX 970 and 980Ti GPUs and 32 GB RAM. You probably need at least 4GB of GPU
memory and 8GB of RAM to run all the code in this repository.

### Usage
#### Generating the kaggle solution
A commendet bash script to generate our final 2nd place solution can be found 
in `make_kaggle_solution.sh`.

Running all the commands sequentially will probaly take 7 - 10 days on recent
consumer grade hardware. If you have multiple GPUs you can speed things up
by doing training and feature extraction for the two networks in parallel. 
However, due to the computationally heavy data augmentation it may be far less than
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
Example usage:
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
Example usage:
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
Example usage:
```
python transform.py --cnf config/c_128_5x5_32.py --train --test --n_iter 5
python transform.py --cnf config/c_128_5x5_32.py --n_iter 5 --test_dir path/to/other/image/files
python transform.py --test_dir path/to/alternative/test/files
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
  --test_dir TEXT      Override directory with test set images.
  --help               Show this message and exit.
```
##### blend.py
Example usage:
```
python blend.py --per_patient # use configuration in blend.yml
python blend.py --per_patient --feature_file path/to/feature/file
python blend.py --per_patient --test_dir path/to/alternative/test/files

```
```
Usage: blend.py [OPTIONS]

Options:
  --cnf TEXT            Path or name of configuration module.  [default: configs/c_128_4x4_32.py]
  --predict             Make predictions on test set features after training. [default: False]
  --per_patient         Blend features of both patient eyes.  [default: False]
  --features_file TEXT  Read features from specified file.
  --n_iter INTEGER      Number of times to fit and average.  [default: 1]
  --blend_cnf TEXT      Blending configuration file.  [default: blend.yml]
  --test_dir TEXT       Override directory with test set images.
  --help                Show this message and exit.
```

#### Configuration

- The convolutional network configuration is done via the files in the `configs` directory.
- To select different combinations of extracted features for blending edit  `blend.yml`.
- To tune parameters related to blending edit `blend.py` directly.
- To make predictions for a different test set either
  + put the resized images into the `data/test_medium` directory
  + or edit the `test_dir` field in your config file(s) inside the `configs` directory 
  + or pass the `--test_dir /path/to/test/files` argument to `transform.py` and `blend.py`

