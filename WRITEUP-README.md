# Project: Follow Me


[//]: # (Image References)

[image1]: ./misc_images/FCN_Model.JPG
[image2]: ./misc_images/FollowMe.JPG
[image3]: ./misc_images/Drone_Patrol_hero.JPG


## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### 1. Writeup / README

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

- Here it is!

### 2. Network Architecture

For the follow me project, a fully convolutional network (FCN) was impletemented in order to train the semantic segmentation model. The FCN shown in **Figure**  **1**, shows a network of 3 encoder layers, a 1x1 convolutional layer, and 3 decoder layers, provided that the skip connections method was implemented to preserve image information from some of the previous layers and from the input image itself. The output layer of the model is obtained by using softmax activation upon the final layer (decoder 3 / x) as can be seen in the code section bellow. 

This 3 encoder-decoder architecture was selected through testing during the Semantic Segmentation lab, and thought more layers or deeper architecture might help to generate a better trained model. 

![alt text][image1]
###### **Figure**  **1** : Network Architecture


	def fcn_model(inputs, num_classes):
    
		# TODO Add Encoder Blocks. 
		# Remember that with each encoder layer, the depth of your model (the number of filters) increases.
		#Encoder 1
		enc_bloc1 = encoder_block(input_layer=inputs, filters=32, strides=2)
		#Encoder 2
		enc_bloc2 = encoder_block(input_layer=enc_bloc1, filters=64, strides=2)
		#Encoder 3
		enc_bloc3 = encoder_block(input_layer=enc_bloc2, filters=128, strides=2)
		
		# TODO Add 1x1 Convolution layer using conv2d_batchnorm().
		conv_1x1 = conv2d_batchnorm(input_layer=enc_bloc3, filters=128, kernel_size=1, strides=1)
		
		# TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
		#Decoder 1 from encoder 2
		dec_bloc2 = decoder_block(small_ip_layer=conv_1x1, large_ip_layer=enc_bloc2, filters=64)
		#Decoder 2 from encoder 1
		dec_bloc1 = decoder_block(small_ip_layer=dec_bloc2, large_ip_layer=enc_bloc1, filters=32)
		#Decoder 3/Final Decoder from inputs
		x = decoder_block(small_ip_layer=dec_bloc1, large_ip_layer=inputs, filters=num_classes)
		
		# The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
		return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)


### 3. Neural Network Paramaters  
In the code section below, the chosen network parameters can be seen. Some of these parameters were chosen based on the data that the model would use to train, while others were intuitively selected and through iterations. As seen in Table 1, the hyperparameters were modified by iterations until the final score of 40.6% was achieved


	learning_rate = 0.001
	batch_size = 128
	num_epochs = 10
	steps_per_epoch = 128
	validation_steps = 50
	workers = 10

The first iteration of training was performed locally to test out feasibility without using AWS and wasting credits, this training provided weights with ~36% accuracy.

On the first AWS GPU training, the learning rate was reduced by half, the amount of images downloaded (around 4131 images) were considered to choose the batch_size and steps_per_epoch ~ the method initially used was images/2*batch_size = steps_per_epoch ~ using 64 as initial batch_size resulting in steps_per_epoch ~= 32. While the num_epochs and workers values were increase to improve learning and speeding training process, respectively. These hyperparameters produced weights of 37% accuracy.

The second AWS training iteration, only batch_size and steps_per_epoch were modified, to be doubled from the previous iteration. This produced weights of 39% accuracy.

The third AWS iteration, learning_rate was the only modified parameter, changing from 0.005 to 0.001. This improved accuracy slightly further to 39.4%.

On the final AWS training iteration, the steps_per_epoch were modified, doubled from 64 to 128. Retraining the model with the latest hyperparameters provided an accuracy of 40.6%


Hyperparameters | Local Training | AWS Training 1 | AWS Training 2  | AWS Training 3 | AWS Training 4 
--- | --- | --- | --- | --- | ---
learning_rate | 0.01 | 0.005 | 0.005 | 0.001 | 0.001
batch_size | 64 | 64 | 128 | 128 | 128
num_epochs | 4 | 10 | 10 | 10 | 10
steps_per_epoch | 200 | 32 | 64 | 64 | 128
validation_steps | 50 | 50 | 50 | 50 | 50
workers | 2 | 10 | 10 | 10 | 10

![alt text][image2]
###### **Figure**  **2** : Model Tested in Sim


### 4. Fully Connected layer vs. 1x1 Convolutional layer
#### Fully Connected Layer 
From what I could understand, in a fully connected layer each neuron is connected to every neuron from the layer that came before it, with each connection having its own weight. When the only requirement is classifying something; convolutional layers are fed into a fully connected layer, these are implemented after flattening its preceding convolutional layer to a 2D tensor.  

#### 1x1 Convolutional layer 
The understanding is that a 1x1 convolutional layer replaces a fully connected layer when creating a fully convolutional network (FCN), to preserve a 4D tensor instead of flattening to 2D. This will allow spatial information to be preserved and it will be able to be decoded with transposed layers afterwards. Also provides further depth and parameters to the network, while being computationaly inexpensive.


### 5. Encoders / Decoders
Encoders and decoders are the two parts that normally used to create a FCN. Encoders are a series of convolutional layers, which purpose is to extract features from the images (or any other input). The encoder layers extract features in stages, from input to deeper and deeper layers, obtaining more finer features the deeper they go. However, the encoder loses sight of the bigger picture and some information is lost, which is where a decoder comes into play. The decoder upscales the output of the encoder; using transposed convolutions (or deconvolutions) it brings the features back to the bigger picture (back to the same size of the original image). This setup results in the segmentation or prediction of each individual pixel in the original image. One problem than may arise from this type of model is that upscaling and upsampling solely from the encoder layers stems from ultimately losing some information. However, this can be prevented to some extent by using the skip-connections method, which will allow the decoder use information from multiple resolutions, resulting in more precise segmentation decisions. 


### 6. Other uses for Model and Data. Further improvements
#### Other uses
From what I understand this model might be able to be repurposed to follow another object such as a car, if training data for car classification is obtained. The FCN model itself seems well structured to be trained on different objects, all it requires is new training and validation data for that specific object. 

#### Further Improvements
As seen on **Figure**  **3**, segmentation results might be improved on when Hero is far. Maybe better resolution images or even more training data with far away hero. This might help train the model better to recognize the hero when far. 

Also, to improve accuracy, more training data could be collected. For this project I used training data provided, and although I managed to achieve 40.6% (0.406) IoU final score, a better score might be attainable with better training data, and possibly different hyperparameters. More epochs and a lower learning rate might also help increase the accuracy of the model. 

![alt text][image3]
###### **Figure**  **3** : Drone Patrol with Hero











