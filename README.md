# Behavioral Cloning

*Self-driving Car Nanodegree at Udacity.*

------------


**Goals:**

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

------------

**Tools:**
- [CarND-Behavioral-Cloning-P3](https://github.com/udacity/CarND-Behavioral-Cloning-P3 "CarND-Behavioral-Cloning-P3")
- [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit "CarND-Term1-Starter-Kit")
- [Conda](https://conda.io/docs/using/envs.htmlhttp:// "Conda")
- [Python](https://www.python.org "Python")
- [Keras](https://keras.io "Keras")
- [Simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip "Simulator")

------------


#### Model Architecture:

For this project I used the NVidia architecture, which offers a simple and reliable solution for training self driving cars models. The Behavioral Cloning module in the Self-driving car nanodegree provides the code that implements the model architecture in Keras, however futher information was acquired in the following docs:

- [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf "End to End Learning for Self-Driving Cars")
- [devblogs.nvidia.com](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ "deep-learning-self-driving-cars")

As per the NVidia documentation, my model consists of the following layers:


| Layer        | Description  |        
| ------------- |:-------------:
| Input     | 66x200x3 | 
| **Normalization**      | lambda x: x/127.5 -1   |
| **Convolution 5x5**      | depth: 24, border: valid, subsample: 2x2, activation: Elu    |
| **Convolution 5x5**      | depth: 36, border: valid, subsample: 2x2, activation: Elu    |
| **Convolution 5x5**      | depth: 48, border: valid, subsample: 2x2, activation: Elu    |
| **Convolution 3x3**      | depth: 64, border: valid, subsample: 2x2, activation: Elu    |
| **Convolution 3x3**      | depth: 64, border: valid, subsample: 2x2, activation: Elu    |
| **Dropout**      | keep_prob: 0.2  |
| Flatten   |   |
| **Dense** | ouput: 100, activation: Elu, L2: 0.000001 |
| **Dense** | ouput: 50, activation: Elu, L2: 0.0000001 |
| **Dense** | ouput: 10, activation: Elu, L2: 0.0000001 |
| **Dense** | ouput: 1 |

A dropout layer was added after the final convolution layer in order to add some noise in the fully connect layers. Also, L2 regularization was added to the fully connected layers to penalize higher weigths. These two methods were used to reduce overfitting and kept with small values - dropout: 2 and l2: 0.0000001. Higher values had a undesirable effect of keeping high training and validation losses with the current dataset.

The full implementation of the NVidia model can be found in `model.py`

------------


#### Training Data:

Training data was acquired by running the [Simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip "Simulator") on track one. Since track one has fewers curvers in comparison to straight lines, and is more biased towards left angles, the following strategy was used to get a balanced data:

- 3-5 complete laps
- 5-8 laps recording only cuvers in forward and backward order
- 2-3 complete laps in backwards order
- 1-2 recovery laps to teach the model how to recover from road boundaries

A total number of [**39600**](https://drive.google.com/open?id=0BwpbZUTOeyiIdGxPN0p1SlZ0WmM "**39600**") images was acquired in this process.

------------


#### Training Strategy:

The model was training for 100 epochs with a batch size of 32 and 2640 samples per epoch (train data divided by 12). These numbers achieved a good balance of tranining speed and accuracy. Bellow is the graph showing traing and validation loss: 

![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/loss.png)

In the graph above the training stopped around the 50th epoch due to Keras early stopping callback, which has patience set to 5. Though, it's possible to achieve better results by removing or increasing the patience.

Before feeding the images to the model a preprocessing step was included in order to crop unnecessary features of the image and resize it to the NVidia model input shape specification. The simulator is `Height: 160, Width: 320 and 3 Channels`. After preprocessing each image had shape `Height: 66, Width: 200 and 3 Channels`. 

Note: The same preprocessing step needs to be applied to the images in the simulator autonomous mode. This can be done in `drive.py`

example of original and preprocessed images:

###### original image
![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/original.jpg)
###### pre-processed image
![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/preprocessed.jpg)

In order to reduce overfitting, data augmentation was applied to the training dataset by randomly shifting the image in horizontal and vertical directions, and changing the brighteness to simulate day an night conditions. This data augmentation flow is based on NVidia's own flow, which involves shifting and rotating the images randomly.

###### shiftted image
![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/translated.jpg)
###### brightness change (afternoon)
![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/afternoon.jpg)
###### brightness change (night)
![alt text](https://github.com/ismalakazel/carnd-behavioral-cloning/blob/master/examples/night.jpg)

Section 3 and 5.2 in the [NVidia paper](NVidia paper "NVidia paper") explains how they went about collecting and augmenting data to train the model.


#### Result:

[Track one](https://www.youtube.com/watch?v=yPswtGSkGLQ)

#### Future Work:

Test the model on track two.

