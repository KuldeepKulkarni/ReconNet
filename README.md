# [ReconNet: Non-Iterative Reconstruction of Images From Compressively Sensed Measurements](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kulkarni_ReconNet_Non-Iterative_Reconstruction_CVPR_2016_paper.pdf)
[Kuldeep Kulkarni](http://www.public.asu.edu/~kkulkar1/), [Suhas Lohit](https://www.linkedin.com/in/suhaslohit), [Pavan Turaga](http://www.public.asu.edu/~pturaga/Welcome.html), [Ronan Kerviche](http://wp.optics.arizona.edu/trifimaging/trif-imaging-fellows/ronan-kerviche/), [Amit Ashok](http://fp.optics.arizona.edu/ashoka/HomeSite/Home.html),
[The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016] (http://cvpr2016.thecvf.com/), pp. 449-458. 

Project Page: http://www.public.asu.edu/~kkulkar1/reconnet.htm
## Introduction:
ReconNet is a non-iterative and an extremely fast algorithm to reconstruct images from compressively sensed (CS) random measurements. In the paper, we show significant improvements in reconstruction results (both in terms of PSNR and time complexity) over state-of-the-art iterative CS reconstruction algorithms at various measurement rates. The code provided here helps one to reproduce some of the results presented in the paper.

## Citation (BibTex):
If you are using this code, please cite the following paper.
```
@InProceedings{Kulkarni_2016_CVPR,
author = {Kulkarni, Kuldeep and Lohit, Suhas and Turaga, Pavan and Kerviche, Ronan and Ashok, Amit},
title = {ReconNet: Non-Iterative Reconstruction of Images From Compressively Sensed Measurements},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2016}
}
```
## System Requirements:
This software has been tested on Linux 64-bit system.
### Prerequisites
1. [Caffe](http://caffe.berkeleyvision.org/) including matcaffe
2. MATLAB (Tested on 2014b and 2015b)

## Installing ReconNet:
1. Download and unzip ReconNet-master.zip. Place "ReconNet-master" folder in "examples" directory of your caffe installation.

2. Run ./ReconNet-master/setup_training.sh to download and unzip the training and validation datasets ([Source] (http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)). This step is required only if you want to train the caffemodels. (see below for more on training.)

## Reconstruction using ReconNet:
The pre-trained models for measurement rates of 0.25, 0.1, 0.04 and 0.01 and the corresponding measurement matrices are provided in the ./ReconNet-master/test/caffemodels and ./ReconNet-master/test/phi directories respectively. ./ReconNet-master/test_images contains the test images (downloaded from two sources: http://dsp.rice.edu/software/DAMP-toolbox and http://see.xidian.edu.cn/faculty/wsdong/NLR_Exps.htm) used to produce the results in tables 1 and 2, and figures 3 and 4 in the paper. The reconstructed images for the different measurement rates are provided in ./ReconNet/test/reconstruction_results.

To reproduce the results, first select a measurement rate by editing ./ReconNet/test/test_everything.m accordingly and run test_everything.m. Please note the time complexity results provided in table 2 of the paper were produced using a NVIDIA GTX 980 GPU and hence may not necessarily match if you are using a different GPU. 

## Training models:
### Training ReconNet from scratch:
The network definition and parameters of the initial random weights of the network are provided in ./ReconNet-master/train/ReconNet_arch.prototxt and the optimization parameters in ./ReconNet-master/train/ReconNet_solver.prototxt.

1. Run generate_train.m from ./ReconNet-master/train/ directory in MATLAB to sample the image patches of size 33 by 33 which act as the training labels, and the corresponding  random Gaussian measurements (using a measurement matrix in ./ReconNet-master/phi directory) which act as training inputs for the network. The training inputs and labels will be saved in hdf5 format in ./ReconNet-master/train/train.h5. Similarly run ./ReconNet-master/train/generate_test.m to generate the validation set which will be saved in test.h5.

2. Open the terminal and run ./ReconNet-master/train/train.sh. A directory to save the caffemodels is created before the training begins.

## Contact:
Kuldeep Kulkarni, (kkulkar1@asu.edu)

## Acknowledgements:
Our training code is inspired by the [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) code.
