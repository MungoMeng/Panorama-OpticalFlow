//
//  PixFlow.hpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h> 

#include "opencv2/core/cuda.hpp"

#include "util.hpp"
#include "PixFlow_GPU.hpp"

namespace optical_flow {
    using namespace std;
    using namespace cv;
    using namespace cv::detail;
    using namespace util;

    OpticalFlowInterface_GPU* makeOpticalFlowByName_GPU(const string flowAlgName) {
        //若选择pixflow_low模式，则进行以下操作
        if (flowAlgName == "pixflow_low") {
            static const float kPyrScaleFactor                  = 0.9f;
            static const float kSmoothnessCoef                  = 0.001f;
            static const float kVerticalRegularizationCoef      = 0.01f;
            static const float kHorizontalRegularizationCoef    = 0.01f;
            static const float kGradientStepSize                = 0.5f;
            static const float kDownscaleFactor                 = 0.5f;
            static const float kDirectionalRegularizationCoef   = 0.0f;
            static const int   MaxPercentage                    = 0;
            return new PixFlow_GPU(
                                  kPyrScaleFactor,
                                  kSmoothnessCoef,
                                  kVerticalRegularizationCoef,
                                  kHorizontalRegularizationCoef,
                                  kGradientStepSize,
                                  kDownscaleFactor,
                                  kDirectionalRegularizationCoef,
                                  MaxPercentage
                                  );
        }
        //若选择pixflow_search_20模式，则进行以下操作
        if (flowAlgName == "pixflow_search_20") {
            static const float kPyrScaleFactor                  = 0.9f;
            static const float kSmoothnessCoef                  = 0.001f;
            static const float kVerticalRegularizationCoef      = 0.01f;
            static const float kHorizontalRegularizationCoef    = 0.01f;
            static const float kGradientStepSize                = 0.5f;
            static const float kDownscaleFactor                 = 0.5f;
            static const float kDirectionalRegularizationCoef   = 0.0f;
            static const int   MaxPercentage                    = 20;
            return new PixFlow_GPU(
                                   kPyrScaleFactor,
                                   kSmoothnessCoef,
                                   kVerticalRegularizationCoef,
                                   kHorizontalRegularizationCoef,
                                   kGradientStepSize,
                                   kDownscaleFactor,
                                   kDirectionalRegularizationCoef,
                                   MaxPercentage
                                   );
        }
        //若flowAlgName为不可识别的字符串，则抛出错误信息
        throw VrCamException("unrecognized flow algorithm name: " + flowAlgName);
    }

    //定义在cuda上运行时需要用到的inline函数
    __device__ inline float getPixBilinear32FExtend(const cuda::PtrStepSz<float> img, float x, float y) {
        x                 = fminf(img.cols - 2.0f, fmaxf(0.0f, x));
        y                 = fminf(img.rows - 2.0f, fmaxf(0.0f, y));
        const int x0      = int(x);
        const int y0      = int(y);
        const float xR    = x - float(x0);
        const float yR    = y - float(y0);
        const float f00   = img(y0,x0);
        const float f01   = img(y0+1,x0);
        const float f10   = img(y0,x0+1);
        const float f11   = img(y0+1,x0+1);
        const float a1    = f00;
        const float a2    = f10 - f00;
        const float a3    = f01 - f00;
        const float a4    = f00 + f11 - f10 - f01;
        const float result = a1 + a2 * xR + a3 * yR + a4 * xR * yR;
        return result;
    }

    __device__ inline float errorFunction(
                                   const cuda::PtrStepSz<float> I0,
                                   const cuda::PtrStepSz<float> I1,
                                   const cuda::PtrStepSz<float> alpha0,
                                   const cuda::PtrStepSz<float> alpha1,
                                   const cuda::PtrStepSz<float> I0x,
                                   const cuda::PtrStepSz<float> I0y,
                                   const cuda::PtrStepSz<float> I1x,
                                   const cuda::PtrStepSz<float> I1y,
                                   const int x,
                                   const int y,
                                   const cuda::PtrStepSz<float2> flow,
                                   const cuda::PtrStepSz<float2> blurredFlow,
                                   const float2 flowDir,
                                   const float smoothnessCoef,
                                   const float verticalRegularizationCoef,
                                   const float horizontalRegularizationCoef) {
        const float matchX      = x + flowDir.x;
        const float matchY      = y + flowDir.y;
        const float i0x         = I0x(y, x);
        const float i0y         = I0y(y, x);
        const float i1x         = getPixBilinear32FExtend(I1x, matchX, matchY);
        const float i1y         = getPixBilinear32FExtend(I1y, matchX, matchY);
        const float flowDiffx  = fabsf(blurredFlow(y, x).x - flowDir.x);
        const float flowDiffy  = fabsf(blurredFlow(y, x).y - flowDir.y);
        const float smoothness  = sqrtf(flowDiffx*flowDiffx+flowDiffy*flowDiffy);  
        float err = sqrtf(fabsf((i0x - i1x) * (i0x - i0y) + (i0y - i1y) * (i0y - i1y)))
            + smoothness * smoothnessCoef
            + verticalRegularizationCoef * fabsf(flowDir.y) / float(I0.cols)
            + horizontalRegularizationCoef * fabsf(flowDir.x) / float(I0.cols);

        return err;
    }
        
    __device__ inline float2 errorGradient(
                                     const cuda::PtrStepSz<float> I0,
                                     const cuda::PtrStepSz<float> I1,
                                     const cuda::PtrStepSz<float> alpha0,
                                     const cuda::PtrStepSz<float> alpha1,
                                     const cuda::PtrStepSz<float> I0x,
                                     const cuda::PtrStepSz<float> I0y,
                                     const cuda::PtrStepSz<float> I1x,
                                     const cuda::PtrStepSz<float> I1y,
                                     const cuda::PtrStepSz<float2> flow,
                                     const cuda::PtrStepSz<float2> blurredFlow,
                                     const int x,
                                     const int y,
                                     const float currErr,
                                     const float kGradEpsilon,
                                     const float smoothnessCoef,
                                     const float verticalRegularizationCoef,
                                     const float horizontalRegularizationCoef) {
            
        float2 dx;
        dx.x = flow(y, x).x + kGradEpsilon; 
        dx.y = flow(y, x).y + 0.0f;
        float2 dy;
        dy.x = flow(y, x).x + 0.0f; 
        dy.y = flow(y, x).y + kGradEpsilon;
            
        const float fx = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow,dx,
                                       smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
        const float fy = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow,dy,
                                       smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
            
        float2 result;
        result.x = (fx - currErr) / kGradEpsilon;
        result.y = (fy - currErr) / kGradEpsilon;
        return result;
    }

    //定义sweep核，对每一个像素点进行并行的sweep操作
    __global__ void Sweep_Kernel(const cuda::PtrStepSz<float> I0,
                                      const cuda::PtrStepSz<float> I1,
                                      const cuda::PtrStepSz<float> alpha0,
                                      const cuda::PtrStepSz<float> alpha1,
                                      const cuda::PtrStepSz<float> I0x,
                                      const cuda::PtrStepSz<float> I0y,
                                      const cuda::PtrStepSz<float> I1x,
                                      const cuda::PtrStepSz<float> I1y,
                                      cuda::PtrStepSz<float2> flow,
                                      const cuda::PtrStepSz<float2> blurredFlow,
                                      const float kUpdateAlphaThreshold,
                                      const float smoothnessCoef,
                                      const float verticalRegularizationCoef,
                                      const float horizontalRegularizationCoef,
                                      const float gradientStepSize,
                                      const float kGradEpsilon){
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < flow.cols && y < flow.rows && alpha0(y, x) > kUpdateAlphaThreshold && alpha1(y, x) > kUpdateAlphaThreshold) {
            float currErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow(y, x),
                                          smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);

            //对4邻域的点计算err值，若发现更好的匹配点，则用其替换当前点的值
            if (y > 0){
                float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow(y - 1, x),
                                                  smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
                if (proposalErr < currErr) {
                    flow(y, x) = flow(y - 1, x);
                    currErr = proposalErr;
                }
            }
            if (x > 0){
                float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow(y, x - 1),
                                                  smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
                if (proposalErr < currErr) {
                    flow(y, x) = flow(y, x - 1);
                    currErr = proposalErr;
                }
            }
            if (y < flow.rows -1){
                float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow(y + 1, x),
                                                  smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
                if (proposalErr < currErr) {
                    flow(y, x) = flow(y + 1, x);
                    currErr = proposalErr;
                }
            }
            if (x < flow.cols - 1){
                float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow(y, x + 1),
                                                  smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
                if (proposalErr < currErr) {
                    flow(y, x) = flow(y, x + 1);
                    currErr = proposalErr;
                }
            }

            float2 errorGrad = errorGradient(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y,flow, blurredFlow, x, y,currErr,
                               kGradEpsilon,smoothnessCoef,verticalRegularizationCoef,horizontalRegularizationCoef);
            flow(y, x).x = flow(y, x).x - gradientStepSize * errorGrad.x;
            flow(y, x).y = flow(y, x).y - gradientStepSize * errorGrad.y;
        }
    }

    //在GPU加速模式下使用的patchMatchPropagationAndSearch函数
    void PixFlow_GPU::patchMatchPropagationAndSearch(
                                        const Mat& I0,
                                        const Mat& I1,
                                        const Mat& alpha0,
                                        const Mat& alpha1,
                                        Mat& flow,
                                        DirectionHint hint) {
            
        // image gradients
        Mat I0x, I0y, I1x, I1y;
        const int kSameDepth = -1; // same depth as source image
        const int kKernelSize = 1;
        Sobel(I0, I0x, kSameDepth, 1, 0, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
        Sobel(I0, I0y, kSameDepth, 0, 1, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
        Sobel(I1, I1x, kSameDepth, 1, 0, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
        Sobel(I1, I1y, kSameDepth, 0, 1, kKernelSize, 1, 0.0f, BORDER_REPLICATE);
            
        // blur gradients
        const cv::Size kGradientBlurSize(kGradientBlurKernelWidth, kGradientBlurKernelWidth);
        GaussianBlur(I0x, I0x, kGradientBlurSize, kGradientBlurSigma);
        GaussianBlur(I0y, I0y, kGradientBlurSize, kGradientBlurSigma);
        GaussianBlur(I1x, I1x, kGradientBlurSize, kGradientBlurSigma);
        GaussianBlur(I1y, I1y, kGradientBlurSize, kGradientBlurSigma);
        
        if (flow.empty()) {
            // initialize to all zeros
            flow = Mat::zeros(I0.size(), CV_32FC2);
            // optionally look for a better flow
            if (MaxPercentage > 0 && hint != DirectionHint::UNKNOWN) {
                PixFlow_GPU::adjustInitialFlow(I0, I1, alpha0, alpha1, flow, hint);
            }
        }

        // blur flow. we will regularize against this
        Mat blurredFlow;
        GaussianBlur(
                     flow,
                     blurredFlow,
                     cv::Size(kBlurredFlowKernelWidth, kBlurredFlowKernelWidth),
                     kBlurredFlowSigma);
        const cv::Size imgSize = I0.size();

        //载入数据到GPU设备端
        cuda::GpuMat g_I0(I0);
        cuda::GpuMat g_I1(I1);
        cuda::GpuMat g_alpha0(alpha0);
        cuda::GpuMat g_alpha1(alpha1);
        cuda::GpuMat g_I0x(I0x);
        cuda::GpuMat g_I0y(I0y);
        cuda::GpuMat g_I1X(I1x);
        cuda::GpuMat g_I1y(I1y);
        cuda::GpuMat g_flow(flow);
        cuda::GpuMat g_blurredFlow(blurredFlow);

        //定义线程块的大小及数量
        dim3 threadsPerBlock(32,32);
        dim3 blocksPerGrid((flow.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (flow.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        //循环调用sweep核，多次进行并行的sweep操作
        for(int i = 0; i < 10; i++)
        Sweep_Kernel<<<blocksPerGrid, threadsPerBlock>>>(g_I0,
                                                         g_I1,
                                                         g_alpha0,
                                                         g_alpha1,
                                                         g_I0x,
                                                         g_I0y,
                                                         g_I1X,
                                                         g_I1y,
                                                         g_flow,
                                                         g_blurredFlow,
                                                         kUpdateAlphaThreshold,
                                                         smoothnessCoef,
                                                         verticalRegularizationCoef,
                                                         horizontalRegularizationCoef,
                                                         gradientStepSize,
                                                         kGradEpsilon);

        //数据由GPU设备载出到CPU
        g_flow.download(flow);

        medianBlur(flow, flow, kMedianBlurSize);
        PixFlow_GPU::lowAlphaFlowDiffusion(alpha0, alpha1, flow);
    }

    //定义lowAlphaFlowDiffusion核，对每一个像素点进行并行的操作
    __global__ void lowAlphaFlowDiffusion_Kernel(const cuda::PtrStepSz<float> alpha0, 
                                                 const cuda::PtrStepSz<float> alpha1, 
                                                 cuda::PtrStepSz<float2> flow, 
                                                 const cuda::PtrStepSz<float2> blurredFlow){
        int X = blockDim.x * blockIdx.x + threadIdx.x;
        int Y = blockDim.y * blockIdx.y + threadIdx.y;
        
        if(X < flow.cols && Y < flow.rows){
            const float a0 = alpha0(Y, X);
            const float a1 = alpha1(Y, X);
            const float diffusionCoef = 1.0f - a0 * a1;
            flow(Y, X).x =diffusionCoef * blurredFlow(Y, X).x
                        + (1.0f - diffusionCoef) * flow(Y, X).x;
            flow(Y, X).y =diffusionCoef * blurredFlow(Y, X).y
                        + (1.0f - diffusionCoef) * flow(Y, X).y;
        }
    }

    //在GPU加速模式下使用的lowAlphaFlowDiffusion函数
    void PixFlow_GPU::lowAlphaFlowDiffusion(const Mat& alpha0, const Mat& alpha1, Mat& flow) {
        Mat blurredFlow;
        GaussianBlur(
                     flow,
                     blurredFlow,
                     Size(kBlurredFlowKernelWidth, kBlurredFlowKernelWidth),
                     kBlurredFlowSigma);

        //数据载入到GPU设备
        cuda::GpuMat g_alpha0(alpha0);
        cuda::GpuMat g_alpha1(alpha1);
        cuda::GpuMat g_blurredFlow(blurredFlow);
        cuda::GpuMat g_flow(flow);

        //定义线程块的大小及数量
        dim3 threadsPerBlock(32,32);
        dim3 blocksPerGrid((flow.cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (flow.rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        //用于调试，输出相关信息
        //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid.x*blocksPerGrid.y, threadsPerBlock.x*threadsPerBlock.y);

        //调用lowAlphaFlowDiffusion核，进行并行计算
        lowAlphaFlowDiffusion_Kernel<<<blocksPerGrid, threadsPerBlock>>>(g_alpha0, g_alpha1, g_flow, g_blurredFlow);

        //数据由GPU设备载出到CPU
        g_flow.download(flow);
    }
}
