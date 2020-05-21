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

using namespace std;
using namespace cv;
using namespace util;
using namespace optical_flow;
using namespace stitch_tools;

//定义需要传入的参数
string FLAGS_test_dir = "/home/rachel/Programs/test/test1";
string FLAGS_top_img = "top.tif";
string FLAGS_flow_alg = "pixflow_search_20";

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
    //记录程序开始的时刻
    double StartTime = getCurrTimeSec();
    
    //检查命令传入的参数是否为空
    requireArg(FLAGS_test_dir, "test_dir");
    requireArg(FLAGS_top_img, "top_img");
    requireArg(FLAGS_flow_alg, "flow_alg");
    
    //定义Mat变量，并先载入位于顶部的图片
    Mat colorImageL,colorImageR,FinalResult;
    Mat colorImageT = imreadExceptionOnFail(FLAGS_test_dir + "/" + FLAGS_top_img, -1);
    //检查载入图片是否有alpha通道。若没有，则将其变为4通道图像
    if (colorImageT.type() == CV_8UC3) {    cvtColor(colorImageT, colorImageT, CV_BGR2BGRA); }
    
    //进入循环，依次读入名称为序号1-5的图片。
    //首先从1图开始与先前载入的顶部图片进行融合，得到结果
    //然后继续读入后续图片，并与上一轮循环得到的融合结果图进行融合
    for(int i = 1; i <= 5; i++){
        //记录该次循环的开始时刻
        double StartTime = getCurrTimeSec();
        
        //若为第一轮循环，则以读入的顶部图作为“右”图
        //否则，则将上一轮循环得到的循环结果作为“右”图
        if(i == 1) colorImageR = colorImageT;
        else       colorImageR = FinalResult;
        
        //读入待融合的“左”图
        colorImageL = imreadExceptionOnFail(FLAGS_test_dir + "/" + char(i+48) + ".tif", -1);
        //检查载入图片是否有alpha通道。若没有，则将其变为4通道图像
        if (colorImageL.type() == CV_8UC3) {    cvtColor(colorImageL, colorImageL, CV_BGR2BGRA); }
        
        //定义Stitchtools类对象
        Stitchtools Stools;
        //输入“左”图与“右”图，计算出两图重合区域、重合区域分布Map、融合系数Blend
        Stools.prepare(colorImageL, colorImageR);
        
        //用于调试，输出中间结果
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/OverlappedR.png", Stools.getOverlappedR());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/OverlappedL.png", Stools.getOverlappedL());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/Blend.png", 255*Stools.getBlend());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/Map.png", Stools.getMap());
        
        //定义Mat变量，并从Stitchtools类对象中取出下一步需要的参数
        Mat overlappedL = Stools.getOverlappedL();
        Mat overlappedR = Stools.getOverlappedR();
        Mat blend = Stools.getBlend();
        
        //定义NovelViewGenerator类指针并初始化为NovelViewGeneratorAsymmetricFlow型对象，模式为FLAGS_flow_alg
        NovelViewGenerator* novelViewGen =new NovelViewGeneratorAsymmetricFlow(FLAGS_flow_alg);
        //输入两图重合区域，计算出重合区域的光流图
        //double StartTime1 = getCurrTimeSec();
        novelViewGen->prepare(overlappedL, overlappedR);
        //double EndTime1 = getCurrTimeSec();
        //cout << "OpticalFlow Finished!"<<"RUNTIME (sec) = " << (EndTime1 - StartTime1)<<endl;

        //用于调试，生成可视化光流图并输出中间结果
        //buildvisualizations(novelViewGen->getFlowLtoR(),novelViewGen->getFlowRtoL(),colorImageL,colorImageR);
        
        //对NovelViewGeneratorAsymmetricFlow型对象载入后续操作中需要用到的Blend
        novelViewGen->setBlend(blend);
        //定义Mat型变量，用于储存重合部分的融合结果
        Mat novelViewMerged = Mat();
        novelViewGen->generateNovelView(novelViewMerged);
        
        //用于调试，输出中间结果
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/mergedmiddle.png", novelViewMerged);
        
        //对Stitchtools类对象载入后面步骤中需要用到的novelViewMerged
        Stools.setMergedmiddle(novelViewMerged);
        //将“左”原图与“右”原图已经融合的中间部分拼成一张完整的图
        Stools.Gather();
        FinalResult = Stools.getFinalResult();
        
        //若为最后一轮循环则输出最终结果，否则输出中间结果用于调试
        if(i == 5)
            imwriteExceptionOnFail(FLAGS_test_dir + "/" + "FinalResult.png", FinalResult);
        else
            imwriteExceptionOnFail(FLAGS_test_dir + "/" + "ProcessResult"+ char(i+48) +".png", FinalResult);
        
        //清空NovelViewGenerator类型的指针
        delete novelViewGen;
        //记录该次循环的结束时刻，并输出融合两张图片的运行时间
        double EndTime = getCurrTimeSec();
        cout << "Part"<<i<<" Finished!"<<"RUNTIME (sec) = " << (EndTime - StartTime)<<endl;
    }
    
    //记录程序结束的时刻，并输出程序总的运行时间
    double EndTime = getCurrTimeSec();
    cout << "TotalRunTime (sec) = " << (EndTime - StartTime)<<endl;
    return EXIT_SUCCESS;
}
