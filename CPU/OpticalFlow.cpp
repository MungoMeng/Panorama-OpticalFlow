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
        Point2f flowDir = flow.at<Point2f>(y, x);
        
        int srcx = int(x + flowDir.x * t);
        if(srcx > srcImage.cols - 1) srcx = srcx - srcImage.cols;
        if(srcx < 0)                 srcx = srcx + srcImage.cols;
        
        int srcy = int(y + flowDir.y * t);
        if(srcy > srcImage.rows - 1) srcy = srcImage.rows - 1;
        if(srcy < 0)                 srcy = 0;
        
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
        
        Mat blendImage(imageL.size(), CV_8UC4);
        for (int y = 0; y < imageL.rows; ++y) {
            for (int x = 0; x < imageL.cols; ++x) {
                float blendR;
                float blendL;
                blendR = blend.at<float>(y,x);
                blendL = 1- blendR;

                const Vec4b colorL = NovelViewUtil::generateNovelViewPoint(imageL,flowRtoL,blendR,x,y);
                const Vec4b colorR = NovelViewUtil::generateNovelViewPoint(imageR,flowLtoR,blendL,x,y);
                
                Vec4b colorMixed;

                if (colorL[3] == 0 || colorR[3] == 0 ) {
                    colorMixed = Vec4b(0, 0, 0, 0);
                }
                else {
                    const Point2f fLR = flowLtoR.at<Point2f>(y, x);
                    const Point2f fRL = flowRtoL.at<Point2f>(y, x);

                    static const float kColorDiffCoef = 10.0f;
                    static const float kSoftmaxSharpness = 10.0f;
                    static const float kFlowMagCoef = 100.0f;
                    
                    const float flowMagLR = sqrtf(fLR.x * fLR.x + fLR.y * fLR.y) / float(imageL.cols);
                    const float flowMagRL = sqrtf(fRL.x * fRL.x + fRL.y * fRL.y) / float(imageL.cols);
                    
                    const float colorDiff =
                    (std::abs(colorL[0] - colorR[0]) +
                     std::abs(colorL[1] - colorR[1]) +
                     std::abs(colorL[2] - colorR[2])) / 255.0f;
                    const float deghostCoef = tanhf(colorDiff * kColorDiffCoef);
                    
                    const float alphaL = colorL[3] / 255.0f;
                    const float alphaR = colorR[3] / 255.0f;
                    
                    const double expL =
                    exp(kSoftmaxSharpness * blendL * alphaL * (1.0 + kFlowMagCoef * flowMagRL));
                    const double expR =
                    exp(kSoftmaxSharpness * blendR * alphaR * (1.0 + kFlowMagCoef * flowMagLR));
                    
                    const double sumExp = expL + expR + 0.00001;
                    const float softmaxL = float(expL / sumExp);
                    const float softmaxR = float(expR / sumExp);
                    
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
        outNovelViewMerged = NovelViewUtil::combineNovelViews(
                                                              imageL,
                                                              imageR,
                                                              flowLtoR, flowRtoL,
                                                              Blend);
    }
    
    void NovelViewGeneratorAsymmetricFlow::prepare(
                                                   const Mat& colorImageL,
                                                   const Mat& colorImageR){
        
        imageL = colorImageL.clone();
        imageR = colorImageR.clone();
        
        //extended to increase the continuity between lift and right boundary
        Mat n_imageL,n_imageR;
        Mat Lpart,Rpart;
        
        int length = imageL.cols/20;
        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);
        
        Lpart = imageL(Range(0,imageL.rows),Range(0,length));
        Rpart = imageL(Range(0,imageL.rows),Range(imageL.cols-length,imageL.cols));
        warpPerspective(imageL,n_imageL,shftMat,Size(imageL.cols+2*length,imageL.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(n_imageL,Rect(0,0,length,imageL.rows)));
        Lpart.copyTo(Mat(n_imageL,Rect(n_imageL.cols-length,0,length,imageL.rows)));
        
        Lpart = imageR(Range(0,imageL.rows),Range(0,length));
        Rpart = imageR(Range(0,imageL.rows),Range(imageL.cols-length,imageL.cols));
        warpPerspective(imageR,n_imageR,shftMat,Size(imageL.cols+2*length,imageL.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(n_imageR,Rect(0,0,length,imageL.rows)));
        Lpart.copyTo(Mat(n_imageR,Rect(n_imageL.cols-length,0,length,imageL.rows)));
        
        OpticalFlowInterface* flowAlg = makeOpticalFlowByName(flowAlgName);
        
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
        
        delete flowAlg;
        
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
