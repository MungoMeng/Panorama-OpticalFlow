//
//  main.cpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include <iostream>
#include <string>
#include <vector>

#include "util.hpp"
#include "OpticalFlow.hpp"
#include "StitchTool.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace util;
using namespace optical_flow;
using namespace stitch_tools;

//定义需要传入的参数
DEFINE_string(test_dir,                   "",     "path to dir with test files");
DEFINE_string(flow_alg,                   "",     "optical flow algorithm to use(pixflow_low or pixflow_search_20)");

//用于调试，生成可视的光流图并储存
void buildvisualizations(const Mat flowLtoR, const Mat flowRtoL, const Mat ImageL, const Mat ImageR){
    //调用OpticalFlow.hpp中定义的函数生成可视化光流图
    Mat flowVisLtoR                = visualizeFlowAsGreyDisparity(flowLtoR);
    Mat flowVisRtoL                = visualizeFlowAsGreyDisparity(flowRtoL);
    Mat flowVisLtoRColorWheel      = visualizeFlowColorWheel(flowLtoR);
    Mat flowVisRtoLColorWheel      = visualizeFlowColorWheel(flowRtoL);
    Mat flowVisLtoRColorWithLines  = visualizeFlowAsVectorField(flowLtoR, ImageL);
    Mat flowVisRtoLColorWithLines  = visualizeFlowAsVectorField(flowRtoL, ImageR);
    
    cvtColor(flowVisRtoL,                 flowVisRtoL,                CV_GRAY2BGRA);
    cvtColor(flowVisLtoR,                 flowVisLtoR,                CV_GRAY2BGRA);
    cvtColor(flowVisLtoRColorWheel,       flowVisLtoRColorWheel,      CV_BGR2BGRA);
    cvtColor(flowVisRtoLColorWheel,       flowVisRtoLColorWheel,      CV_BGR2BGRA);
    
    //将3幅图拼接成一幅图像进行输出
    Mat horizontalVisLtoR = stackHorizontal(
                                            vector<Mat>({flowVisLtoR, flowVisLtoRColorWheel, flowVisLtoRColorWithLines}));
    Mat horizontalVisRtoL = stackHorizontal(
                                            vector<Mat>({flowVisRtoL, flowVisRtoLColorWheel, flowVisRtoLColorWithLines}));
    
    imwriteExceptionOnFail(
                           FLAGS_test_dir + "/disparity/LtoR_" + FLAGS_flow_alg + ".png",
                           horizontalVisLtoR);
    
    imwriteExceptionOnFail(
                           FLAGS_test_dir + "/disparity/RtoL_" + FLAGS_flow_alg +".png",
                           horizontalVisRtoL);
}

int main(int argc, char** argv) {
    //初始化
    initOpticalFlow(argc, argv);
    //记录程序开始的时刻
    double StartTime = getCurrTimeSec();
    
    //检查命令传入的参数是否为空
    requireArg(FLAGS_test_dir, "test_dir");
    requireArg(FLAGS_flow_alg, "flow_alg");
    
    //对4张图片进行预裁剪
    Mat colorImage1,colorImage2,colorImage3,colorImage4;
    colorImage1 = imreadExceptionOnFail(FLAGS_test_dir + "/1.tif", -1);
    if (colorImage1.type() == CV_8UC3) {    cvtColor(colorImage1, colorImage1, CV_BGR2BGRA); }
    colorImage2 = imreadExceptionOnFail(FLAGS_test_dir + "/2.tif", -1);
    if (colorImage2.type() == CV_8UC3) {    cvtColor(colorImage2, colorImage2, CV_BGR2BGRA); }
    colorImage3 = imreadExceptionOnFail(FLAGS_test_dir + "/3.tif", -1);
    if (colorImage3.type() == CV_8UC3) {    cvtColor(colorImage3, colorImage3, CV_BGR2BGRA); }
    colorImage4 = imreadExceptionOnFail(FLAGS_test_dir + "/4.tif", -1);
    if (colorImage4.type() == CV_8UC3) {    cvtColor(colorImage4, colorImage4, CV_BGR2BGRA); }
    
    for (int x = 0; x < colorImage1.cols; ++x) {
        for (int y = 0; y < colorImage1.rows; ++y){
            if( !colorImage1.at<Vec4b>(colorImage1.rows/2,x)[3] )
                colorImage1.at<Vec4b>(y,x) = Scalar(0,0,0,0);
            if( !colorImage2.at<Vec4b>(colorImage1.rows/2,x)[3] )
                colorImage2.at<Vec4b>(y,x) = Scalar(0,0,0,0);
            if( !colorImage3.at<Vec4b>(colorImage1.rows/2,x)[3] )
                colorImage3.at<Vec4b>(y,x) = Scalar(0,0,0,0);
            if( !colorImage4.at<Vec4b>(colorImage1.rows/2,x)[3] )
                colorImage4.at<Vec4b>(y,x) = Scalar(0,0,0,0);
        }
    }
    
    Mat colorImageL,colorImageR,FinalResult;
    colorImageL = colorImage1 + colorImage3;
    colorImageR = colorImage2 + colorImage4;
    
    //选择是否裁剪掉图片最下方的三脚架部分，0.95表示裁掉最下方5%
    //colorImageR = colorImageR(Range(0,0.95*colorImageR.rows),Range(0,colorImageR.cols)).clone();
    //colorImageL = colorImageL(Range(0,0.95*colorImageL.rows),Range(0,colorImageL.cols)).clone();
    
    //用于调试，输出中间结果
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/colorImageL.png", colorImageL);
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/colorImageR.png", colorImageR);
    
    //定义Stitchtools类对象
    Stitchtools Stools;
    //输入“左”图与“右”图，计算出两图重合区域、重区域分布Map、融合系数Blend
    Stools.prepare(colorImageL, colorImageR);
    
    //用于调试，输出中间结果
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/OverlappedR.png", Stools.getOverlappedR());
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/OverlappedL.png", Stools.getOverlappedL());
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/Blend.png", 255*Stools.getBlend());
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/Map.png", Stools.getMap());
    
    //定义Mat变量，并从Stitchtools类对象中取出下一步需要的参数
    Mat overlappedL = Stools.getOverlappedL();
    Mat overlappedR = Stools.getOverlappedR();
    Mat blend = Stools.getBlend();
    
    //定义NovelViewGenerator类指针并初始化为NovelViewGeneratorAsymmetricFlow型对象，模式为FLAGS_flow_alg
    NovelViewGenerator* novelViewGen =new NovelViewGeneratorAsymmetricFlow(FLAGS_flow_alg);
    //输入两图重合区域，计算出重合区域的光流图
    novelViewGen->prepare(overlappedL, overlappedR);
    
    //对NovelViewGeneratorAsymmetricFlow型对象载入后续操作中需要用到的Blend
    novelViewGen->setBlend(blend);
    //定义Mat型变量，用于储存重合部分的融合结果
    Mat novelViewMerged = Mat();
    novelViewGen->generateNovelView(novelViewMerged);
    
    //用于调试，输出中间结果
    imwriteExceptionOnFail(FLAGS_test_dir + "/process/mergedmiddle.png", novelViewMerged);
    
    //对Stitchtools类对象载入后面步骤中需要用到的novelViewMerged
    Stools.setMergedmiddle(novelViewMerged);
    //将“左”原图与“右”原图已经融合的中间部分拼成一张完整的图
    Stools.Gather();
    FinalResult = Stools.getFinalResult();
    
    //输出最终结果
    imwriteExceptionOnFail(FLAGS_test_dir + "/FinalResult.png", FinalResult);
    
    //清空NovelViewGenerator类型的指针
    delete novelViewGen;
    
    //记录程序结束的时刻，并输出程序总的运行时间
    double EndTime = getCurrTimeSec();
    cout << "TotalRunTime (sec) = " << (EndTime - StartTime)<<endl;
    return EXIT_SUCCESS;
}
