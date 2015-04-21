# Kaggle Diabetic Retinopathy Detection

### Dependencies
Install dependencies,
```
pip install -r requirements.txt
```
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

### TODO
- Figure out what else needs to be done for preprocessing and normalization in
  order to actually learn something useful.
- Tweak initialization of weights and learning rate.
- Make predictions on test set.
- Can somehow use the non-differentiable quadratic weighted kappa metric?
- Reduce Memory footprint by not loading entire training set into memory.
- ...
