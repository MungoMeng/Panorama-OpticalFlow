//
//  OpticalFlow.hpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#ifndef OpticalFlow_hpp
#define OpticalFlow_hpp

#include <stdio.h>

#include <string>
#include <vector>

#include "util.hpp"
#include "PixFlow.hpp"

namespace optical_flow {
    using namespace std;
    using namespace cv;
    using namespace cv::detail;
    using namespace util;
    
    //用于光流法融合图像重合区域
    struct NovelViewUtil {
        //给定原图srcImage，目标图到原图的光流图flow，偏移系数t（范围0-1）
        //计算返回目标图中的点(x,y)处的像素值
        static Vec4b generateNovelViewPoint(
                                            const Mat& srcImage,
                                            const Mat& flow,
                                            const double t,
                                            const int x,
                                            const int y);
        
        //给定左图imageL、右图imageR、左图到右图的光流图flowLtoR、右图到左图的光流图flowRtoL、全局融合系数Blend
        //生成返回重合区域的融合结果图
        static Mat combineNovelViews(
                                     const Mat& imageL,
                                     const Mat& imageR,
                                     const Mat& flowLtoR,
                                     const Mat& flowRtoL,
                                     const Mat& blend);
    };
    
    //用于计算光流图的类，为基类，定义虚函数
    class NovelViewGenerator {
    public:
        virtual ~NovelViewGenerator() {};
        
        //用于对后续的操作进行预准备
        //输入左右图后，生成LtoR于RtoL的光流图
        virtual void prepare(
                             const Mat& colorImageL,
                             const Mat& colorImageR) = 0;
        
        //调用NovelViewUtil结构体中的函数完成融合区域的生成
        virtual void generateNovelView(Mat& outNovelViewMerged) = 0;
        
        //用于调试，外部读取类内部参数
        virtual Mat getFlowLtoR() { return Mat(); }
        virtual Mat getFlowRtoL() { return Mat(); }
        //外部设置类内部参数
        virtual void setBlend(const Mat& blend) = 0;
    };
    
    //用于同时计算左右图相对于对方的光流偏移量
    //FtoR于RtoL的光流偏移量为非对称的
    class NovelViewGeneratorAsymmetricFlow : public NovelViewGenerator {
    public:
        string flowAlgName;              //用于表示选用计算光流的模式（pixflow_low or pixflow_search_20）
        Mat imageL, imageR;              //用于储存左右图
        Mat flowLtoR, flowRtoL;          //用于储存LtoR与RtoL的光流偏移量
        Mat Blend;                       //用于储存融合系数
        
        //构造函数，输入flowAlgName
        NovelViewGeneratorAsymmetricFlow(const string flowAlgName) : flowAlgName(flowAlgName) {}
        
        ~NovelViewGeneratorAsymmetricFlow() {}
        
        void prepare(
                     const Mat& colorImageL,
                     const Mat& colorImageR);
        
        void generateNovelView(Mat& outNovelViewMerged);
        
        Mat getFlowLtoR() { return flowLtoR; }
        Mat getFlowRtoL() { return flowRtoL; }
        void setBlend(const Mat& blend) { Blend = blend.clone();  }
    };
    
    //用于生成光流的可视化灰度图，灰度表示光流强度
    Mat visualizeFlowAsGreyDisparity(const Mat& flow);
    
    //用于生成在原图的基础上标识出的光流向量图
    Mat visualizeFlowAsVectorField(const Mat& flow, const Mat& image);
    
    //用于生成光流是可视化色度图，色度表示光流方向
    Mat visualizeFlowColorWheel(const Mat& flow);
}

#endif /* OpticalFlow_hpp */
