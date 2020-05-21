#include <cuda_runtime.h>
#include "opencv2/core/cuda.hpp"

#include "StitchTool.hpp"

namespace stitch_tools{
    using namespace std;
    using namespace cv;

    __global__ void countblend_Kernel(const cuda::PtrStepSz<unsigned char> Map, 
                                      cuda::PtrStepSz<float> blend, 
                                      cuda::PtrStepSz<float> MergedDis,
                                      const int length){
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        
        if(x < blend.cols && y < blend.rows){

             if(Map(y,x+length) == 100)      blend(y,x) = 0;

             else if(Map(y,x+length) == 50)  blend(y,x) = 1;

             else if(Map(y,x+length) == 150){

                int step;
                if(blend.cols <= blend.rows) step = blend.cols/200; else step = blend.rows/200;

                float minLdis = 10*blend.cols,minRdis = 10*blend.cols;
        
                //search in 8 directions
                const float sqrt2 = 1.4142;
                for(int i = 0; i < blend.cols/2; i = i + step){        

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

                blend(y,x) = minLdis/(minRdis+minLdis);
                if(minLdis<minRdis) MergedDis(y,x+length)=minLdis;
                else                MergedDis(y,x+length)=minRdis;
             }
             else                            blend(y,x) = 0.5;
        }
    }

    void Stitchtools::GenerateBlend_GPU(){

        Mat blend = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));
        
        //extended to increase continuity between life and right boundary
        int length = Map.cols/5;

        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);

        Mat Lpart,Rpart;
        Lpart = Map(Range(0,Map.rows),Range(0,length));
        Rpart = Map(Range(0,Map.rows),Range(Map.cols-length,Map.cols));
        warpPerspective(Map,Map,shftMat,Size(Map.cols+2*length,Map.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(Map,Rect(0,0,length,Map.rows)));
        Lpart.copyTo(Mat(Map,Rect(Map.cols-length,0,length,Map.rows)));

        MergedDis = Mat(Map.rows, Map.cols, CV_32FC1,Scalar(0));

        cuda::GpuMat g_blend(blend);
        cuda::GpuMat g_Map(Map);
        cuda::GpuMat g_MergedDis(MergedDis);

        dim3 threadsPerBlock(32,32);
        dim3 blocksPerGrid((blend.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (blend.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);

        countblend_Kernel<<<blocksPerGrid, threadsPerBlock>>>(g_Map, g_blend, g_MergedDis, length);

        g_MergedDis.download(MergedDis);
        g_blend.download(blend);

        Map = Map(Range(0,Map.rows),Range(length,Map.cols-length)).clone();
        MergedDis = MergedDis(Range(0,MergedDis.rows),Range(length,MergedDis.cols-length)).clone();

        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;

        //smoothen
        for (int y = 0; y+step < ImageL.rows; y=y+step) {
            for (int x = 0; x+step < ImageL.cols; x=x+step) {
                if(MergedDis.at<float>(y,x) > step){
                    Mat temp = blend(Range(y,y+step),Range(x,x+step));
                    blur(temp,temp,Size(ImageL.rows/130,ImageL.rows/130));
                }
            }
        }
        blur(blend,blend,Size(ImageL.rows/400,ImageL.rows/400));

        Blend = blend.clone();
    }
}
