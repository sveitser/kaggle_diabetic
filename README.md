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
heavy data augmentation it may be far less than twice as fast.

You can also obtain a quadratic weighted kappa score of 0.840 on the private
leaderboard by just training the 4x4 network and by doing on only 
50 feature extracting iterations with the weights that gave you the best
validation scores. The entire ensemble achieves a score of 0.845.

#### Scripts
All these python scripts can be invoced with `-h` to display a brief help
message.
- `convert.py` crops and resizes images
- `train_nn.py` trains convolutional networks
- `transform.py` extracts features from trained convolutional networks
- `blend.py` blends features, optionally using inputs from both patient eyes

```bash
python NAME_OF_SCRIPT.py --help            # display command line parameters
# examples
python convert.py --directory data/train   # resize training set images
python convert.py --directory data/test    # resize test set images
python train_nn.py --cnf config/large.py   # train a conv net for feature extraction.
python fit.py --cnf config/large.py        # extract features and fit regression model
python predict.py --cnf config/large.py    # make predictions on test set
python transform.py --cnf config/large.py  # feature extraction
python boost.py                            # boost extracted features with xgboost
```
The neural network code is mainly based on [this tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Notes
#### Regression vs. Classification
By default treats the problem as regression problem (layers in
```config/large.py```) with mean squared error or
as classification problem (layers in ```config/clf.py```) with softmax output
layer and conversion to labels by weighted average over predicted
probabilities. To use classification the ```REGRESSION``` variable in
```definitions.py``` currently needs to be chaneged to ```False```. Results are
about the same for both approaches so far.

#### Simultaneous Left and Right Eye Prediction
- To train for both eyes simultaneously the ```nn.DoubleIterator``` can be used
instead of ```nn.SingleIterator``` but not everything is implemented yet.
- There is a network config in ```config/double.py```.
- The variable  ```definitions.N_TARGETS``` needs to be set to 2.
- Results are about the same as using each eye individually so far.
- Predicting the test set for the separate eyes at the end isn't
implemented.

#### Preprocessing
The ```convert.py``` script blurs the original image to figure out what the
background color is and then crops the foreground tightly before resizing the
images. Some images have artifacts that it might make sense to remove.

#### Augmentation
- File ```augment.py```.
- Replace zero pixels of each channel by channel mean upon loading image to 
  reduce the strength of the features which simply represent the shape of the 
  eye.
- Random uniform rotations.

### TODO (in arbitrary order)
- ~~Augmentation (random cropping, rotation, shearing, ...)~~
- Tweak initialization of weights and learning rate.
- ~~Make predictions on test set.~~
- Can we somehow use the non-differentiable quadratic weighted kappa metric to
  train the neural network?
- ~~Reduce Memory footprint by not loading entire training set into memory.~~
- ~~Use last layer weights to train sklearn estimator.~~
- ~~Evaluate quadratic weighted kappa metric.~~
- Retrain on full training set after initial training with held out validation
  set.
- ~~Use entire image of eye instead of center crop.~~
- Try converting pixels into polar coordinates(?)
- ~~Balance classes.~~ Can now be selected as a ratio of applied balancing
  in [0, 1].
- ~~Training on two eyes simultaneously.~~
- Tweak network for predicting both eyes simultaneously.
- ~~L2 regularization.~~
- Use a better loss function better suited for ordinal classification than mean
  squared error.
- ...

*working on
