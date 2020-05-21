# Panorama Stitching based on Asymmetric Bidirectional Optical Flow
we proposed a panorama stitching algorithm based on asymmetric bidirectional optical flow. 
This algorithm expects multiple photos captured by fisheye lens cameras as input, and then, 
through the proposed algorithm, these photos can be merged into a high-quality 360-degree spherical panoramic image. 
For photos taken from a distant perspective, the parallax among them is relatively small, 
and the obtained panoramic image can be nearly seamless and undistorted. 
For photos taken from a close perspective or with a relatively large parallax, 
a seamless though partially distorted panoramic image can also be obtained. Besides, 
with the help of Graphics Processing Unit (GPU), this algorithm can complete the whole stitching process at a very fast speed:
typically, it only takes less than 30s to obtain a panoramic image of 9000-by-4000 pixels, 
which means our panorama stitching algorithm is of high value in many real-time applications.

## Workflow
![workflow](https://github.com/MungoMeng/Panorama_OpticalFlow/blob/master/Figure/Workflow.png)
