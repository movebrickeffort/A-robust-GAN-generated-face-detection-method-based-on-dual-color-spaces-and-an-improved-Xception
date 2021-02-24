# A-robust-GAN-generated-face-detection-method-based-on-dual-color-spaces-and-an-improved-Xception
## Overview
#### In this paper, some experimental studies on detecting the post-processed GAN-generated face images find that: (a) both the luminance component and chrominance components play an important role; (b) the RGB and YCbCr color spaces achieve better performance than the HSV and Lab color spaces. Therefore, in order to enhance the robustness, both the luminance component and chrominance components of dual-color spaces (RGB and YCbCr) are considered to utilize color information effectively. Besides, convolutional block attention module and multi-layer feature aggregation module are introduced into the Xception model to enhance its representation power of the feature map and aggregate multi-layer features, respectively. Experimental results demonstrate that our method outperforms some existing methods, especially in robustness against different types of post-processing operations, such as JPEG compression, Gaussian blurring, Gamma correction, and so on.

## Prerequisites
#### Ubuntu 18.04
#### NVIDIA GPU+CUDA CuDNN (CPU mode may also work, but untested)
#### Install Tensorflow and dependencies

## Training and Test Details
#### When you train a RGB or YCbCr single-stream model, you should change the input (input.py or input_ycbcr.py) in train.py. The corresponding part should also be modified during testing. When testing the dual-stream model, the RGB image and its YCbCr image should be input together.

## Related Works
#### [1]Chollet F, “Xception: Deep learning with depthwise separable convolutions,” In: Proceedings of the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR2017), pp. 1251-1258, 2017.
