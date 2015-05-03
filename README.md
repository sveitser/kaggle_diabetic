# Kaggle Diabetic Retinopathy Detection

### Dependencies
Install python dependencies.
```
pip install --upgrade -r requirements.txt
```
Note that this includes a fork of the nolearn package that adds
a ```transform``` method.

It might also require NVIDIA CUDA to run.

Extract train/test images to ```data/train``` and ```data/test``` respectively.

```bash
python convert.py --directory data/train # resize training set images
python convert.py --directory data/test  # resize test set images
python train_nn.py      # train a conv net for feature extraction.
python fit.py           # extract features and fit regression model
python predict.py       # make predictions on test set
```

The neural network code is mainly based on [this tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Notes
#### Regression vs. Classification
By default treats the problem as regression problem (layers in
```config/clf.py```) with mean squared error or
as classification problem (layers in ```config/clf.py```) with softmax output
layer and conversion to labels by weighted average over predicted
probabilities. To use classification the ```REGRESSION``` variable in
```definitions.py``` currently needs to be chaneged to ```False```. Results are
about the same for both approaches so far.

#### Simultaneous Left and Right Eye Prediction
- To train for both eyes simultaneously the ```nn.DoubleIterator``` can be used
instead of `nn.SingleIterator``` but not everything is implemented yet.
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
- Random uniform rotations (currently disabled).
- Random channel multiplication (function ```rgb_mix``, currently disabled).

### TODO (in arbitrary order)
- Augmentation (random cropping, rotation, shearing, ...)
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
