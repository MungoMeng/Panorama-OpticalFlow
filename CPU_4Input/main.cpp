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

DEFINE_string(test_dir,                   "",     "path to dir with test files");
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
    initOpticalFlow(argc, argv);

    double StartTime = getCurrTimeSec();
    
    requireArg(FLAGS_test_dir, "test_dir");
    requireArg(FLAGS_flow_alg, "flow_alg");
    
    //pre-crop 4 input photos
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
    
    //colorImageR = colorImageR(Range(0,0.95*colorImageR.rows),Range(0,colorImageR.cols)).clone();
    //colorImageL = colorImageL(Range(0,0.95*colorImageL.rows),Range(0,colorImageL.cols)).clone();
    
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/colorImageL.png", colorImageL);
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/colorImageR.png", colorImageR);
    
    Stitchtools Stools;
    Stools.prepare(colorImageL, colorImageR);
    
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/OverlappedR.png", Stools.getOverlappedR());
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/OverlappedL.png", Stools.getOverlappedL());
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/Blend.png", 255*Stools.getBlend());
    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/Map.png", Stools.getMap());
    
    Mat overlappedL = Stools.getOverlappedL();
    Mat overlappedR = Stools.getOverlappedR();
    Mat blend = Stools.getBlend();
    
    NovelViewGenerator* novelViewGen =new NovelViewGeneratorAsymmetricFlow(FLAGS_flow_alg);
    novelViewGen->prepare(overlappedL, overlappedR);
    
    novelViewGen->setBlend(blend);
    Mat novelViewMerged = Mat();
    novelViewGen->generateNovelView(novelViewMerged);

    //imwriteExceptionOnFail(FLAGS_test_dir + "/process/mergedmiddle.png", novelViewMerged);
    
    Stools.setMergedmiddle(novelViewMerged);
    Stools.Gather();
    FinalResult = Stools.getFinalResult();
    
    imwriteExceptionOnFail(FLAGS_test_dir + "/FinalResult.png", FinalResult);
    
    delete novelViewGen;
    
    double EndTime = getCurrTimeSec();
    cout << "TotalRunTime (sec) = " << (EndTime - StartTime)<<endl;
    return EXIT_SUCCESS;
}
