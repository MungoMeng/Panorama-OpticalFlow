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

DEFINE_string(test_dir,                   "",     "path to dir with test files");
DEFINE_string(top_img,                    "",     "path to top image  (relative to test_dir)");
DEFINE_string(flow_alg,                   "",     "optical flow algorithm to use(pixflow_low or pixflow_search_20)");

void buildvisualizations(const Mat flowLtoR, const Mat flowRtoL, const Mat ImageL, const Mat ImageR){

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
    double StartTime = getCurrTimeSec();
    
    requireArg(FLAGS_test_dir, "test_dir");
    requireArg(FLAGS_top_img, "top_img");
    requireArg(FLAGS_flow_alg, "flow_alg");
    
    Mat colorImageL,colorImageR,FinalResult;
    Mat colorImageT = imreadExceptionOnFail(FLAGS_test_dir + "/" + FLAGS_top_img, -1);
    if (colorImageT.type() == CV_8UC3) {    cvtColor(colorImageT, colorImageT, CV_BGR2BGRA); }
    
    for(int i = 1; i <= 5; i++){
        double StartTime = getCurrTimeSec();
        
        if(i == 1) colorImageR = colorImageT;
        else       colorImageR = FinalResult;

        colorImageL = imreadExceptionOnFail(FLAGS_test_dir + "/" + char(i+48) + ".tif", -1);
        if (colorImageL.type() == CV_8UC3) {    cvtColor(colorImageL, colorImageL, CV_BGR2BGRA); }
        
        Stitchtools Stools;
        Stools.prepare(colorImageL, colorImageR);
        
        //用于调试，输出中间结果
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/OverlappedR.png", Stools.getOverlappedR());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/OverlappedL.png", Stools.getOverlappedL());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/Blend.png", 255*Stools.getBlend());
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/Map.png", Stools.getMap());
        
        Mat overlappedL = Stools.getOverlappedL();
        Mat overlappedR = Stools.getOverlappedR();
        Mat blend = Stools.getBlend();
        
        NovelViewGenerator* novelViewGen =new NovelViewGeneratorAsymmetricFlow(FLAGS_flow_alg);
        //double StartTime1 = getCurrTimeSec();
        novelViewGen->prepare(overlappedL, overlappedR);
        //double EndTime1 = getCurrTimeSec();
        //cout << "OpticalFlow Finished!"<<"RUNTIME (sec) = " << (EndTime1 - StartTime1)<<endl;

        //buildvisualizations(novelViewGen->getFlowLtoR(),novelViewGen->getFlowRtoL(),colorImageL,colorImageR);
        
        novelViewGen->setBlend(blend);
        Mat novelViewMerged = Mat();
        novelViewGen->generateNovelView(novelViewMerged);
        
        //imwriteExceptionOnFail(FLAGS_test_dir + "/process"+ char(i+48) + "/mergedmiddle.png", novelViewMerged);
        
        Stools.setMergedmiddle(novelViewMerged);
        Stools.Gather();
        FinalResult = Stools.getFinalResult();
        
        if(i == 5)
            imwriteExceptionOnFail(FLAGS_test_dir + "/" + "FinalResult.png", FinalResult);
        else
            imwriteExceptionOnFail(FLAGS_test_dir + "/" + "ProcessResult"+ char(i+48) +".png", FinalResult);
        
        delete novelViewGen;
        double EndTime = getCurrTimeSec();
        cout << "Part"<<i<<" Finished!"<<"RUNTIME (sec) = " << (EndTime - StartTime)<<endl;
    }
    
    double EndTime = getCurrTimeSec();
    cout << "TotalRunTime (sec) = " << (EndTime - StartTime)<<endl;
    return EXIT_SUCCESS;
}
