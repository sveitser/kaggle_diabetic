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
python train_nn.py      # train a conv net and extract features
python fit.py           # fit sklearn estimator
python predict.py       # make predictions on test set
```

The neural network code is mainly based on [this tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Notes
Currently treats the problem as regression problem with mean squared error.

### TODO (in arbitrary order)
- Figure out what else needs to be done for preprocessing and normalization in
  order to actually learn something useful.
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
- Balance classes. *Mathis 
- Check if there is a correlation between left and right eye and maybe train model with both eyes as input and two targets.
- ...

*working on