# Solution Outline

Neural networks were trained using,
 - lasagne https://github.com/Lasagne/Lasagne
 - nolearn https://github.com/dnouri/nolearn

A lot of inspiration and some code was taken from 
the first and second place finishers of the national data science bowl 
competition. Thanks!

## Preprocessing
- Separate eye from background and crop square. In cases where this failed, take 
the largest center square. 
- Resize to 512, 256, 128 pixel squares.

## Augmentation
- 360 degrees rotation, translation, scaling, stretching, Krizhevsky color 
  augmentation

## Network Configuration
            units filter size stride       size
 1                                          448    
 2 Conv        32           5      2        224   
 3 Conv        32           3               224     
 4 MaxPool                  3      2        111     
 5 Conv        64           5      2         56
 6 Conv        64           3                56
 7 Conv        64           3                56
 8 MaxPool                         2         27
 9 Conv       128           3                27
10 Conv       128           3                27  
11 Conv       128           3                27
12 MaxPool                         2         13        *
13 Conv       256           3                13        *
14 Conv       256           3                13        *
15 Conv       256           3                13        *
16 MaxPool                  3      2          6       ** 
17 Conv       512           3                 6       **
18 Conv       512           3                 6       **  
19 RMSPool                  3      2          2
20 Dropout
21 Dense     1024
22 FeatPool   512
23 Dropout
24 Dense     1024
25 FeatPool   512

- Leaky (0.01) rectifier units following each conv and dense layer.
- Nesterov momentum with fixed schedule and 250 iterations.
    + epoch   0 0.003
    + epoch 150 0.0003
    + epoch 220 0.00003
- L2 weight decay factor 0.0005
- Training started with resampling such that all classes were present in equal 
  fractions we then gradually decreased the balancing to final resampling 
  weights of 1 for class 0 and 2 for the other classes.
- Mean squared error objective.

The second configuration is similar like the one above but uses only 4x4 
kernels (slightly different intermedia sizes as a result) and only single conv 
layer with 512 units before the RMSPool layer.

## Training
- 128 px images layers 1 - 11 and 20 to 25.
- 256 px images layers 1 - 15 and 20 to 25 weights for 1 to 11 initialized 
  from above.
- 512 px images all layers weights for 1 - 15 from above.

Models were trained on a GTX 970 and GTX 980Ti with batch sizes small enough 
to fit into memory (usually 32 to 48 for the largest networks and 128 for the 
smaller ones).

## "Per Patient" Blend
Extracted mean and standard deviation of RMSPool layer for 50 pseudo random 
augmentations for three sets of weights (best validation score, best kappa,
final weights).

For each eye used the following as input features for blending
[this_mean, other_mean, this_std, other_std, left_eye_indicator],
standardized all features to have zero mean and unit variance and used this to 
train a network of shape,

Input        8192
Dense          32
FeaturePool    16
Dense          32
FeaturePool    16

with L1 regularization on the first layer and L2 regularization everywhere.

The mean output of the six predictions (2 networks, 3 sets of weights) was then 
thresholded to get the integer levels of the submission.

## Random Notes
- We when using very leaky (0.33) rectifier units we managed to train the
  larger networks directly however the after blending the features for each 
  patients the validation kappa was never quite as good as the one achieved 
  with leaky rectifier units and initialization from smaller nets.
- Data augmentation is tough with large color images. We used 
  https://pypi.python.org/pypi/SharedArray to use multiprocessing without 
  having to pickle the images.
