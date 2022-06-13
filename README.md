# Panorama Stitching based on Asymmetric Bidirectional Optical Flow
We proposed a panorama stitching algorithm based on asymmetric bidirectional optical flow. 
This algorithm expects multiple photos captured by fisheye lens cameras as input, and then, 
through the proposed algorithm, these photos can be merged into 
a high-quality 360-degree spherical panoramic image. 
For photos taken from a distant perspective, the parallax among them is relatively small, 
and the obtained panoramic image can be nearly seamless and undistorted. 
For photos taken from a close perspective or with a relatively large parallax, 
a seamless though partially distorted panoramic image can also be obtained. Besides, 
with the help of Graphics Processing Unit (GPU), this algorithm can complete the whole stitching process at a very fast speed:
typically, it only takes less than 30s to obtain a panoramic image of 9000-by-4000 pixels, 
which means our panorama stitching algorithm is of high value in many real-time applications.  
**For more details, please refer to our paper. [[IEEE](https://ieeexplore.ieee.org/document/9178683)] [[arXiv](https://arxiv.org/abs/2006.01201)].**

## Workflow
![workflow](https://github.com/MungoMeng/Panorama-OpticalFlow/blob/master/Figure/Workflow.png)

Our algorithm can be divided into two stages. The first one is a pre-processing stage that can be implemented by many
open-source packages or existing algorithms, including distortion correction, chromaticity correction, 
and coarse feature-based registration. The second one is an optical flow-based blending stage, 
we iteratively use an image blending algorithm based on asymmetric bidirectional optical flow to finely 
stitch each photo processed through the pre-processing stage.

**This repository only contains the code of the second stage. The first stage can be implemented through many existing packages (e.g. [Hugin](http://hugin.sourceforge.net/)) and is not included in this repository.**

## Result Comparison
![result](https://github.com/MungoMeng/Panorama-OpticalFlow/blob/master/Figure/Result.png)  

## Instruction
Here we provide a GPU version code in the `./GPU` and a CPU-only version code in the `./CPU`. All code is written in **C++**.

### Pre-reqirements
* C++ compiler
* OpenCV-3.20 (Cuda module needs to be open for GPU version code)
* Cuda-8.0 (only for GPU version code)
* gflags, glog

Other versions of OpenCV and Cuda might work as well, but there is no guarantee on it.

### Requirements for input photos
Our algorithm requires the camera to rotate horizontally around a vertical rotation axis, 
or requires multiple cameras around an axis on the same horizontal plane to capture photos at various angles at the same time. 
Besides, it’s also necessary to set a vertical camera to get the scene information in the vertical direction (see Figure below).

![Input_requirement](https://github.com/MungoMeng/Panorama-OpticalFlow/blob/master/Figure/Input_requirement.png)

Before fed into this program, all photos need to be pre-processed with distortion/chromaticity correction and a coarse registration. Then, our code expects 6 pre-processed photos as input. Among them, 1 photo is captured by a vertical camera and is named `top.tif`, other 5 photos are captured by horizontal cameras and are named from `1.tif` to `5.tif`. All these 6 photos should be put into the same directory.  


### Build and Run
Please `cd` to the directory containing all source code, and then use a C++ complier to build the program and get a executable file. For example, for GPU-version code, the commond possibly looks like:  
```
nvcc *.cpp *.cu -o outputfile `pkg-config opencv --cflags --libs` -lglog -lgflags -std=c++11
```
When you get the executable file, you can run it directly with the following parameters:  

* `-test_dir`: specify the directory containing input photos.
* `-top_img`: specify the file name of the top photo such as 'top.tif'.
* `-flow_alg`: specify the mode of calculating optical flow (pixflow_low/pixflow_search_20).

Normally, the code only outputs a panoramic image as the final result. If needed, you can uncomment some lines in the `main.cpp` to get intermediate results of stitching each photo in each iteration.

## Alternative version
We also provide an alternative version code in the `./CPU_4Input`. This program can be used when you have 4 horizontal
photos captured by wide-angle lens cameras. This program can stitch all 4 photos in one pass, so it's much faster than the original version.

## Test data
In the `./Test_data` and `./Test_data_4Input`, we provide some input photos for testing our algorithm/code. 
All input photos have been pre-processed through [Hugin](http://hugin.sourceforge.net/).

## Publication
If this repository helps your work, please kindly cite our paper:
* **Mingyuan Meng, Shaojun Liu, "High-quality Panorama Stitching based on Asymmetric Bidirectional Optical Flow," 5th International Conference on Computational Intelligence and Applications (ICCIA), pp. 118-122, 2020, doi: 10.1109/ICCIA49625.2020.00030. [[IEEE](https://ieeexplore.ieee.org/document/9178683)] [[arXiv](https://arxiv.org/abs/2006.01201)]**
