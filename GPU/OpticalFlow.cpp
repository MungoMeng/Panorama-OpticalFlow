//
//  OpticalFlow.cpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include <iostream>

#include "opencv2/core/cuda.hpp"

#include "OpticalFlow.hpp"

namespace optical_flow {
    using namespace std;
    using namespace cv;
    using namespace cv::detail;
    using namespace util;

    Vec4b NovelViewUtil::generateNovelViewPoint(
                                                const Mat& srcImage,
                                                const Mat& flow,
                                                const double t,
                                                const int x,
                                                const int y) {
        //定义Point2f型变量，储存flow中(x,y)点的偏移量
        Point2f flowDir = flow.at<Point2f>(y, x);
        
        //计算目标图中(x,y)点对应原图中点的横坐标
        int srcx = int(x + flowDir.x * t);
        //若超出原图左（右）边界，则从右（左）边界处往左（右）取点
        if(srcx > srcImage.cols - 1) srcx = srcx - srcImage.cols;
        if(srcx < 0)                 srcx = srcx + srcImage.cols;
        
        //计算目标图中(x,y)点对应原图中点的纵坐标
        int srcy = int(y + flowDir.y * t);
        //若超出原图左（右）边界，则从右（左）边界处往左（右）取点
        if(srcy > srcImage.rows - 1) srcy = srcImage.rows - 1;
        if(srcy < 0)                 srcy = 0;
        
        //后向映射，返回目标图(x,y)处像素值
        Vec4b novelPoint;
        novelPoint = srcImage.at<Vec4b>(srcy,srcx);
        return novelPoint;
    }
    
    Mat NovelViewUtil::combineNovelViews(
                                         const Mat& imageL,
                                         const Mat& imageR,
                                         const Mat& flowLtoR,
                                         const Mat& flowRtoL,
                                         const Mat& blend  ) {
        
        //预定义变量
        Mat blendImage(imageL.size(), CV_8UC4);
        //逐像素点进行遍历，计算生成blendImage
        for (int y = 0; y < imageL.rows; ++y) {
            for (int x = 0; x < imageL.cols; ++x) {
                //由blend中取出对应像素点的值，计算blendL与blendR
                float blendR;
                float blendL;
                blendR = blend.at<float>(y,x);
                blendL = 1- blendR;
                
                //调用generateNovelViewPoint函数
                //生成左（右）图进过blendR（blendL）作为系数偏移后得到该像素点的值
                const Vec4b colorL = NovelViewUtil::generateNovelViewPoint(imageL,flowRtoL,blendR,x,y);
                const Vec4b colorR = NovelViewUtil::generateNovelViewPoint(imageR,flowLtoR,blendL,x,y);
                
                //开始对colorL、colorR进行融合操作
                //定义colorMixed储存融合后该点的像素
                Vec4b colorMixed;
                //若存在colorL或colorR的alpha通道无效区域，则colorMixed也为(0,0,0,0)
                if (colorL[3] == 0 || colorR[3] == 0 ) {
                    colorMixed = Vec4b(0, 0, 0, 0);
                }
                else {
                    //取出光流图中该像素点的偏移量，存入fLR、fRL
                    const Point2f fLR = flowLtoR.at<Point2f>(y, x);
                    const Point2f fRL = flowRtoL.at<Point2f>(y, x);

                    //定义、计算后续操作中相关参数
                    static const float kColorDiffCoef = 10.0f;
                    static const float kSoftmaxSharpness = 10.0f;
                    static const float kFlowMagCoef = 100.0f;
                    //计算光流偏移向量长度与图像边长的比值
                    const float flowMagLR = sqrtf(fLR.x * fLR.x + fLR.y * fLR.y) / float(imageL.cols);
                    const float flowMagRL = sqrtf(fRL.x * fRL.x + fRL.y * fRL.y) / float(imageL.cols);
                    //计算colorDiff用以衡量左右图在该点的差异程度
                    const float colorDiff =
                    (std::abs(colorL[0] - colorR[0]) +
                     std::abs(colorL[1] - colorR[1]) +
                     std::abs(colorL[2] - colorR[2])) / 255.0f;
                    const float deghostCoef = tanhf(colorDiff * kColorDiffCoef);
                    //alpha通道值归一化
                    const float alphaL = colorL[3] / 255.0f;
                    const float alphaR = colorR[3] / 255.0f;
                    //计算expL、expR
                    const double expL =
                    exp(kSoftmaxSharpness * blendL * alphaL * (1.0 + kFlowMagCoef * flowMagRL));
                    const double expR =
                    exp(kSoftmaxSharpness * blendR * alphaR * (1.0 + kFlowMagCoef * flowMagLR));
                    //将expL、expR归一化
                    const double sumExp = expL + expR + 0.00001;
                    const float softmaxL = float(expL / sumExp);
                    const float softmaxR = float(expR / sumExp);
                    
                    //对colorMixed4通道依次赋值
                    colorMixed = Vec4b(
                                       float(colorL[0]) * lerp(blendL, softmaxL, deghostCoef) + float(colorR[0]) * lerp(blendR, softmaxR, deghostCoef),
                                       float(colorL[1]) * lerp(blendL, softmaxL, deghostCoef) + float(colorR[1]) * lerp(blendR, softmaxR, deghostCoef),
                                       float(colorL[2]) * lerp(blendL, softmaxL, deghostCoef) + float(colorR[2]) * lerp(blendR, softmaxR,deghostCoef),
                                       255);
                }
                blendImage.at<Vec4b>(y, x) = colorMixed;
            }
        }
        return blendImage;
    }

    void NovelViewGeneratorAsymmetricFlow::generateNovelView(Mat& outNovelViewMerged) {
        //调用NovelViewUtil中的combineNovelViews函数生成融合图
        outNovelViewMerged = NovelViewUtil::combineNovelViews(
                                                              imageL,
                                                              imageR,
                                                              flowLtoR, flowRtoL,
                                                              Blend);
    }
    
    void NovelViewGeneratorAsymmetricFlow::prepare(
                                                   const Mat& colorImageL,
                                                   const Mat& colorImageR){
        
        //储存左右图
        imageL = colorImageL.clone();
        imageR = colorImageR.clone();

        //考虑全景图片左边界与右边界的连续性，将图片延扩后进行光流计算
        Mat n_imageL,n_imageR;
        Mat Lpart,Rpart;   

        //定义延扩长度
        int length = imageL.cols/20;
        //定义一个向右平移length的平移矩阵
        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);
        
        //对左图进行延拓
        Lpart = imageL(Range(0,imageL.rows),Range(0,length));
        Rpart = imageL(Range(0,imageL.rows),Range(imageL.cols-length,imageL.cols));
        warpPerspective(imageL,n_imageL,shftMat,Size(imageL.cols+2*length,imageL.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(n_imageL,Rect(0,0,length,imageL.rows)));
        Lpart.copyTo(Mat(n_imageL,Rect(n_imageL.cols-length,0,length,imageL.rows)));
        
        //对右图进行延拓
        Lpart = imageR(Range(0,imageL.rows),Range(0,length));
        Rpart = imageR(Range(0,imageL.rows),Range(imageL.cols-length,imageL.cols));
        warpPerspective(imageR,n_imageR,shftMat,Size(imageL.cols+2*length,imageL.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(n_imageR,Rect(0,0,length,imageL.rows)));
        Lpart.copyTo(Mat(n_imageR,Rect(n_imageL.cols-length,0,length,imageL.rows)));
        
        //判断是否存在可用的GPU设备
        bool GPU = true;
        int num_devices = cuda::getCudaEnabledDeviceCount();
        if(num_devices <= 0){
            GPU = false;
            cout << "There is no GPU device." <<endl;
        }
        else
            cout << "There is "<<num_devices<< " GPU device." <<endl;

        int enable_device_id = -1;
        for(int i = 0; i < num_devices; i++){
            cuda::DeviceInfo dev_info(i);
            if(dev_info.isCompatible())   enable_device_id = i;
        }
        if(enable_device_id < 0){
            GPU = false;
            cout << "GPU module isn't built. Use CPU to generate Optical Flow." <<endl;
        }
        else
            cout << "GPU "<<enable_device_id<<" module is built. Use GPU to generate OPtical Flow" <<endl;
        
        //若GPU为true，则选择GPU模式计算光流场
        if(GPU == true){
            //设置选用的GPU模块
            cuda::setDevice(enable_device_id);

            //定义OpticalFlowInterface_GPU指针，并调用makeOpticalFlowByName_GPU函数初始化
            //flowAlgName为计算光流时选用的模式
            OpticalFlowInterface_GPU* flowAlg_GPU = makeOpticalFlowByName_GPU(flowAlgName);
        
            //调用computeOpticalFlow函数计算LtoR与RtoL的光流偏移量
            flowAlg_GPU->computeOpticalFlow_GPU(
                                        n_imageL,
                                        n_imageR,
                                        flowLtoR,
                                        OpticalFlowInterface_GPU::DirectionHint::LEFT);
            flowAlg_GPU->computeOpticalFlow_GPU(
                                        n_imageR,
                                        n_imageL,
                                        flowRtoL,
                                        OpticalFlowInterface_GPU::DirectionHint::RIGHT);
        
            //清空指针
            delete flowAlg_GPU;
        }
        //若GPU为false，则选择普通模式计算光流场
        else{
            //定义OpticalFlowInterface指针，并调用makeOpticalFlowByName函数初始化
            //flowAlgName为计算光流时选用的模式
            OpticalFlowInterface* flowAlg = makeOpticalFlowByName(flowAlgName);
        
            //调用computeOpticalFlow函数计算LtoR与RtoL的光流偏移量
            flowAlg->computeOpticalFlow(
                                        n_imageL,
                                        n_imageR,
                                        flowLtoR,
                                        OpticalFlowInterface::DirectionHint::LEFT);
            flowAlg->computeOpticalFlow(
                                        n_imageR,
                                        n_imageL,
                                        flowRtoL,
                                        OpticalFlowInterface::DirectionHint::RIGHT);
        
            //清空指针
            delete flowAlg;
        }

        //对延扩后生成的光流图进行裁剪
        flowLtoR = flowLtoR(Range(0,flowLtoR.rows),Range(length,flowLtoR.cols-length)).clone();
        flowRtoL = flowRtoL(Range(0,flowRtoL.rows),Range(length,flowRtoL.cols-length)).clone();
    }

    Mat visualizeFlowAsGreyDisparity(const Mat& flow) {
        Mat disparity(flow.size(), CV_32F);
        for (int y = 0; y < flow.rows; ++y) {
            for (int x = 0; x < flow.cols; ++x) {
                disparity.at<float>(y, x) = flow.at<Vec2f>(y, x)[0];
            }
        }
        normalize(disparity, disparity, 0, 255, NORM_MINMAX, CV_32F);
        Mat disaprity8U;
        disparity.convertTo(disaprity8U, CV_8U);
        return disaprity8U;
    }
    
    Mat visualizeFlowAsVectorField(const Mat& flow, const Mat& image) {
        static const int kGridSpacing = 12;
        static const Scalar kGridColor = Scalar(0, 0, 0, 255);
        static const float kArrowLen = 7.0f;
        Mat imageWithFlowLines = image.clone();
        for (int y = kGridSpacing; y < image.rows - kGridSpacing; ++y) {
            for (int x = kGridSpacing; x < image.cols - kGridSpacing; ++x) {
                if (x % kGridSpacing == 0 && y % kGridSpacing == 0) {
                    Point2f fxy = flow.at<Point2f>(y, x);
                    const float mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
                    const static float kEpsilon = 0.1f;
                    fxy /= mag + kEpsilon;
                    line(
                         imageWithFlowLines,
                         Point(x, y),
                         Point(x + fxy.x * kArrowLen, y + fxy.y * kArrowLen),
                         kGridColor,
                         1,
                         CV_AA);
                }
            }
        }
        return imageWithFlowLines;
    }
    
    Mat visualizeFlowColorWheel(const Mat& flow) {
        Mat rgbVis(flow.size(), CV_8UC3);
        const static float kDisplacementScale = 20.0f;
        const float maxExpectedDisplacement =
        float(max(flow.size().width, flow.size().height)) / kDisplacementScale;
        for (int y = 0; y < flow.rows; ++y) {
            for (int x = 0; x < flow.cols; ++x) {
                Point2f flowVec = flow.at<Point2f>(y, x);
                const float mag = sqrt(flowVec.x * flowVec.x + flowVec.y * flowVec.y);
                flowVec /= mag;
                const float brightness = .25f + .75f * min(1.0f, mag / maxExpectedDisplacement);
                const float hue = (atan2(flowVec.y, flowVec.x) + M_PI) / (2.0 * M_PI);
                rgbVis.at<Vec3b>(y, x)[0] = 180.0f * hue;
                rgbVis.at<Vec3b>(y, x)[1] = 255.0f * brightness;
                rgbVis.at<Vec3b>(y, x)[2] = 255.0f * brightness;
            }
        }
        cvtColor(rgbVis,  rgbVis, CV_HSV2BGR);
        return rgbVis;
    }
}
