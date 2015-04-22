# Kaggle Diabetic Retinopathy Detection

### Dependencies
Install python dependencies.
```
pip install -r requirements.txt
```
It might also require NVIDIA CUDA to run.

Extract training images to ```data/train```.

Resize and square crop with,
```
python convert.py 
```
Train a very preliminary regression model (it doesn't really learn anything 
useful yet).

```
python train.py
```
The code is mainly based on [this tutorial](http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/).

### Notes
Currently treats the problem as regression problem.

### TODO
- Figure out what else needs to be done for preprocessing and normalization in
  order to actually learn something useful.
- Augmentation (random cropping, rotation, shearing, ...)
- Tweak initialization of weights and learning rate.
- Make predictions on test set.
- Can somehow use the non-differentiable quadratic weighted kappa metric?
- ~~Reduce Memory footprint by not loading entire training set into memory.~~
- ...
