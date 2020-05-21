//
//  StitchTool.hpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#ifndef StitchTool_hpp
#define StitchTool_hpp

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "util.hpp"

#include <opencv2/opencv.hpp>
#include "opencv2/core/cuda.hpp"

namespace stitch_tools{
    using namespace std;
    using namespace cv;
    
    //用于储存、处理两张经过粗配准对其的图片
    //可以实现图片的重叠区域取出，计算融合系数、合成图片等功能
    class Stitchtools{
    public:
        Mat ImageL,ImageR;                 //用于储存左原图与右原图
        Mat Blend;                         //用于储存融合系数
        Mat OverlappedL,OverlappedR;       //用于储存左右图中重合部分
        Mat Mergedmiddle;                  //用于储存经过光流法融合的中间部分
        Mat Map;                           //用于标记两图在画布中的位置及分布。左原图：100;重合区域：150;右原图：50;
        Mat FinalResult;                   //用于储存最后结果
        Mat MergedDis;                     //中间变量，记录重合区域的点到重合区域边界的距离
        
        Stitchtools() {}
        ~Stitchtools() {}
        
        //输入左右原图，计算得到OverlappedL、OverlappedR、Map、Blend
        void prepare(
                     const Mat& colorImageL,
                     const Mat& colorImageR);
        
        //用于生成Map
        void MatchImages();
        
        //用于将Mergedmiddle于左右原图进行合成得到FinalResult
        void Gather();
        
        //用于生成Blend
        void GenerateBlend();
        void GenerateBlend_GPU();
        
        //用于计算(x,y)点的相似系数blend的值
        float countblend(
                         const int x,
                         const int y);
        
        //从外部调用类内部的部分变量
        Mat getImageL() { return ImageL; }
        Mat getImageR() { return ImageR; }
        Mat getBlend()  { return Blend;  }
        Mat getMap()    { return Map;    }
        Mat getOverlappedL() { return OverlappedL; }
        Mat getOverlappedR() { return OverlappedR; }
        Mat getFinalResult() { return FinalResult; }
        //从外部设置类内部的部分变量
        void setMergedmiddle(const Mat &image) { Mergedmiddle = image.clone();  }
        
    };
}

#endif /* StitchTool_hpp */
