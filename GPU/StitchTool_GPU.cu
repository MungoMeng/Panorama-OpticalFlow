//
//  StitchTool.cpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include <cuda_runtime.h> 
#include "opencv2/core/cuda.hpp"

#include "StitchTool.hpp"

namespace stitch_tools{
    using namespace std;
    using namespace cv;

    //用于计算融合系数Blend的核
    __global__ void countblend_Kernel(const cuda::PtrStepSz<unsigned char> Map, 
                                      cuda::PtrStepSz<float> blend, 
                                      cuda::PtrStepSz<float> MergedDis,
                                      const int length){
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        
        if(x < blend.cols && y < blend.rows){
             //若为左原图的区域，blend = 0
             if(Map(y,x+length) == 100)      blend(y,x) = 0;
             //若为右原图的区域，blend = 1
             else if(Map(y,x+length) == 50)  blend(y,x) = 1;
             //若为重合区域，进行以下运算
             else if(Map(y,x+length) == 150){
                //计算循环步长step
                int step;
                if(blend.cols <= blend.rows) step = blend.cols/200; else step = blend.rows/200;
        
                //初始化minLdis，minRdis的值为极大值
                float minLdis = 10*blend.cols,minRdis = 10*blend.cols;
        
                //进行循环，对该点8个方向进行探索
                //记录与该点最近的左（右）原图边界到该点的距离
                const float sqrt2 = 1.4142;
                for(int i = 0; i < blend.cols/2; i = i + step){        
                    //对8个方向进行检索，若符合条件且minLdis（minRdis）值更小时，替换minLdis（minRdis）值
                    if(x+i+length < Map.cols && Map(y,x+i+length) == 100 && i<minLdis)  minLdis = i;
                    if(x+i+length < Map.cols && Map(y,x+i+length) == 50  && i<minRdis)  minRdis = i;
                    if(x-i+length > 0        && Map(y,x-i+length) == 100 && i<minLdis)  minLdis = i;
                    if(x-i+length > 0        && Map(y,x-i+length) == 50  && i<minRdis)  minRdis = i;
                    if(y+i < Map.rows && Map(y+i,x+length) == 100 && i<minLdis)  minLdis = i;
                    if(y+i < Map.rows && Map(y+i,x+length) == 50  && i<minRdis)  minRdis = i;
                    if(y-i > 0        && Map(y-i,x+length) == 100 && i<minLdis)  minLdis = i;
                    if(y-i > 0        && Map(y-i,x+length) == 50  && i<minRdis)  minRdis = i;
                    if((x+i+length < Map.cols && y+i < Map.rows) && Map(y+i,x+i+length) == 100 && i*sqrt2<minLdis)
                        minLdis = i*sqrt2;
                    if((x+i+length < Map.cols && y+i < Map.rows) && Map(y+i,x+i+length) == 50  && i*sqrt2<minRdis)
                        minRdis = i*sqrt2;
                    if((x-i+length > 0        && y-i > 0       ) && Map(y-i,x-i+length) == 100 && i*sqrt2<minLdis)
                        minLdis = i*sqrt2;
                    if((x-i+length > 0        && y-i > 0       ) && Map(y-i,x-i+length) == 50  && i*sqrt2<minRdis)
                        minRdis = i*sqrt2;
                    if((x+i+length < Map.cols && y-i > 0       ) && Map(y-i,x+i+length) == 100 && i*sqrt2<minLdis)
                        minLdis = i*sqrt2;
                    if((x+i+length < Map.cols && y-i > 0       ) && Map(y-i,x+i+length) == 50  && i*sqrt2<minRdis)
                        minRdis = i*sqrt2;
                    if((x-i+length > 0        && y+i < Map.rows) && Map(y+i,x-i+length) == 100 && i*sqrt2<minLdis)
                        minLdis = i*sqrt2;
                    if((x-i+length > 0        && y+i < Map.rows) && Map(y+i,x-i+length) == 50  && i*sqrt2<minRdis)
                        minRdis = i*sqrt2;
                }
                
                //计算blend
                blend(y,x) = minLdis/(minRdis+minLdis);
                //计算该点到重合区域边界的距离
                if(minLdis<minRdis) MergedDis(y,x+length)=minLdis;
                else                MergedDis(y,x+length)=minRdis;
             }
             //若为空白区域，blend = 0.5
             else                            blend(y,x) = 0.5;
        }
    }

    void Stitchtools::GenerateBlend_GPU(){
        //定义变量并将其初始化为零值矩阵
        Mat blend = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));
        
        //考虑全景图片左边界与右边界的连续性，将Map延扩后进行Blend计算
        //定义延扩长度
        int length = Map.cols/5;

        //定义一个向右平移length的平移矩阵
        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);

        //对Map图进行延拓
        Mat Lpart,Rpart;
        Lpart = Map(Range(0,Map.rows),Range(0,length));
        Rpart = Map(Range(0,Map.rows),Range(Map.cols-length,Map.cols));
        warpPerspective(Map,Map,shftMat,Size(Map.cols+2*length,Map.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(Map,Rect(0,0,length,Map.rows)));
        Lpart.copyTo(Mat(Map,Rect(Map.cols-length,0,length,Map.rows)));
        
        //预定义MergedDis，在计算Blend过程中计算得到
        MergedDis = Mat(Map.rows, Map.cols, CV_32FC1,Scalar(0));
        
        //数据载入到GPU设备端
        cuda::GpuMat g_blend(blend);
        cuda::GpuMat g_Map(Map);
        cuda::GpuMat g_MergedDis(MergedDis);

        //定义线程块的大小及数量
        dim3 threadsPerBlock(32,32);
        dim3 blocksPerGrid((blend.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (blend.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        //用于调试，输出调用信息
        //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);

        //调用countblend核，对每一像素点并行计算Blend值与MergedDis值
        countblend_Kernel<<<blocksPerGrid, threadsPerBlock>>>(g_Map, g_blend, g_MergedDis, length);

        //数据由GPU设备端载出到CPU
        g_MergedDis.download(MergedDis);
        g_blend.download(blend);
        
        //对延扩后生成的Map与MergedDis进行裁剪
        Map = Map(Range(0,Map.rows),Range(length,Map.cols-length)).clone();
        MergedDis = MergedDis(Range(0,MergedDis.rows),Range(length,MergedDis.cols-length)).clone();
        
        //计算step，为循环步长
        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;

        //对step为边长的小正方块进行平滑滤波操作
        //此操作希望仅对重合区域内部进行平滑操作，减小对于重合区域边界的影响
        for (int y = 0; y+step < ImageL.rows; y=y+step) {
            for (int x = 0; x+step < ImageL.cols; x=x+step) {
                //仅对两图重合区域中极力重合区域边界距离大于step的小正方块进行操作
                if(MergedDis.at<float>(y,x) > step){
                    Mat temp = blend(Range(y,y+step),Range(x,x+step));
                    blur(temp,temp,Size(ImageL.rows/130,ImageL.rows/130));
                }
            }
        }
        //对其余部分进行平滑滤波
        blur(blend,blend,Size(ImageL.rows/400,ImageL.rows/400));
        
        //结果存在Blend中
        Blend = blend.clone();
    }
}
