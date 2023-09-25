# BP_PPT_Edge_CNN
[BP_In-Sensor_computing with programmable photonic device developed in UW. Ref [1]] (https://www.nature.com/articles/s41467-022-29171-1)

CNN built by editing the referenced simple-code CNN
Ashutosh Kumar Upadhyay (2016). Convolution Neural Network - simple code - simple to use (https://www.mathworks.com/matlabcentral/fileexchange/59223-convolution-neural-network-simple-code-simple-to-use), MATLAB Central File Exchange.

## Working Principle and Hardware Configurations for In-Sensor Computing
![Picture1](https://github.com/seokhlee/BP_HPT_Edge_CNN/assets/100313451/183c3d60-e195-4594-a0fa-586396ce90d9)
Optoelectronic computation of convolutional kernels. (a) The computing structure of the edge detection, consisting of layers of computation, which resembles convolutional neural networks (CNN). (b) The schematics of optoelectronic convolution computation of infrared image and convolutional kernel. Each pixel of the array is programmed to different photoconductive states and the input image is projected on the device array. (c) Four programmed pixels generate one convolved outcome upon the corresponding four pixels of the image. The outcome of the convolutional result for the vertical kernel is time traced. (d) (e) The computed image after the convolution with (d) vertical kernel and (e) horizontal kernel, with each element indicated on the right column.

![SI_Fig_SI_rev2_meas_schemev2](https://github.com/seokhlee/BP_HPT_Edge_CNN/assets/100313451/0f2ab359-904c-46c1-aa10-54196b0fee67)



## CNN Archetecture Used in the Paper

We pre-trained the simple CNN architecture consisting of two 3x3 Convolutional Kernels, Mean Pooling layer, and a FC layer to classify MNIST '0's and '1's, and obtained the elements of Convolutional Kernels using O1_MNIST_CNN_HPT_train.m

![Picture2](https://github.com/seokhlee/BP_HPT_Edge_CNN/assets/100313451/7bbc590e-f996-496f-b60a-b4d20088a23f)
