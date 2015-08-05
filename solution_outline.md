First I'd like to thank team mate Stephan, the hosts and everyone 
else who joined and congratulate the other winners! 

Below is a brief solution summary while things are fresh before we get
time to do a proper writeup and publish our code.

## Team o_O Solution Summary

Neural networks trained using [lasagne](https://github.com/Lasagne/Lasagne)
and [nolearn](https://github.com/dnouri/nolearn). Both libraries are awesome, 
well documented and easy to get started with.

A lot of inspiration and some code was taken from winners of
the national data science bowl competition. Thanks!

- [≋ Deep Sea ≋](https://github.com/benanne/kaggle-ndsb): 
  data augmentation, RMS pooling, pseudo random augmentation averaging 
  (used for feature extraction here)
- [Happy Lantern Festival](https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code)
  Capacity and coverage analysis of conv nets (it's built into nolearn)

### Preprocessing
- We selected the smallest rectangle that contained the entire eye.
- When this didn't work well (dark images) or the image was already
  almost a square we defaulted to selecting the center square.
- Resized to 512, 256, 128 pixel squares.
- Before the data augmentation we scaled each channel to have zero mean 
  and unit variance.

### Augmentation
- 360 degrees rotation, translation, scaling, stretching (all uniform)
- Krizhevsky color augmentation (gaussian)

These augmentations were always applied.

### Network Configurations
```

                                net A       |          net B
                units   filter stride size  |  filter stride size
     1 Input                           448  |     4           448
     2 Conv        32     5       2    224  |     4     2     224
     3 Conv        32     3            224  |     4           225 
     4 MaxPool            3       2    111  |     3     2     112 
     5 Conv        64     5       2     56  |     4     2      56
     6 Conv        64     3             56  |     4            57
     7 Conv        64     3             56  |     4            56
     8 MaxPool            3       2     27  |     3     2      27
     9 Conv       128     3             27  |     4            28
    10 Conv       128     3             27  |     4            27
    11 Conv       128     3             27  |     4            28
    12 MaxPool            3       2     13  |     3     2      13
    13 Conv       256     3             13  |     4            14
    14 Conv       256     3             13  |     4            13
    15 Conv       256     3             13  |     4            14
    16 MaxPool            3       2      6  |     3     2       6
    17 Conv       512     3              6  |     4             5
    18 Conv       512     3              6  |   n/a           n/a
    19 RMSPool            3       3      2  |     4     2       2
    20 Dropout
    21 Dense     1024
    22 Maxout     512
    23 Dropout
    24 Dense     1024
    25 Maxout     512
```
- Leaky (0.01) rectifier units following each conv and dense layer.
- Nesterov momentum with fixed schedule and 250 epochs.
    + epoch 0: 0.003
    + epoch 150: 0.0003
    + epoch 220: 0.00003
    + For the nets for 256 and 128 pixels images we stopped training already 
      after 200 epochs.
- L2 weight decay factor 0.0005
- Training started with resampling such that all classes were present in equal 
  fractions. We then gradually decreased the balancing after each epoch to
  arrive at final "resampling weights" of 1 for class 0 and 2 for the other 
  classes.
- Mean squared error objective.
- Untied biases.

### Training
- We used 10% of the patients as validation set.
- 128 px images -> layers 1 - 11 and 20 to 25.
- 256 px images -> layers 1 - 15 and 20 to 25. Weights of layer 1 - 11 initialized 
  with weights from above.
- 512 px images -> all layers. Weights of layers 1 - 15 initialized with
  weights from above.

Models were trained on a GTX 970 and a GTX 980Ti (highly recommended) with batch
sizes small enough to fit into memory (usually 24 to 48 for the large networks
and 128 for the smaller ones).

### "Per Patient" Blend
Extracted mean and standard deviation of RMSPool layer for 50 pseudo random 
augmentations for three sets of weights (best validation score, best kappa,
final weights) for net A and B.

For each eye (or patient) used the following as input features for blending,
```
[this_eye_mean, other_eye_mean, this_eye_std, other_eye_std, right_eye_indicator]
```
standardized all features to have zero mean and unit variance and used them to 
train a network of shape,
```
    Input        8193
    Dense          32
    Maxout         16
    Dense          32
    Maxout         16
```
- L1 regularization (2e-5) on the first layer and L2 regularization (0.005) 
  everywhere.
- [Adam Updates](http://arxiv.org/abs/1412.6980) with fixed learning rate
  schedule over 100 epochs.
  + epoch  0: 5e-4
  + epoch 60: 5e-5
  + epoch 80: 5e-6
  + epoch 90: 5e-7
- Mean squared error objective.
- Batches of size 128, replace batch with probability
  + 0.2 with batch sampled such that classes are balanced
  + 0.5 with batch sampled uniformly from all images (shuffled)

The mean output of the six blend networks (conv nets A and B, 3 sets of
weights each) was then thresholded at `[0.5, 1.5, 2.5, 3.5]` to get 
integer levels for submission.

### Notes
I just noticed that we achieved a slightly higher private LB score for a 
submission where we optimized the thresholds on the validation set to
maximize kappa. But this didn't work as well on the public LB and we 
didn't pursue it any further or select it for scoring at the end.

When using very leaky (0.33) rectifier units we managed to train the
networks for 512 px images directly and the MSE and kappa score were comparable
with what we achieved with leaky rectifiers. However after blending the features 
for each patients over pseudo random augmentations the validation and LB kappa 
were never quite as good as what we obtained with leaky rectifier units and
initialization from smaller nets. We went back to using 0.01 leaky rectifiers 
afterwards and gave up on training the larger networks from scratch.
