#ifndef OpticalFlow_hpp
#define OpticalFlow_hpp

#include <stdio.h>

#include <string>
#include <vector>

#include "util.hpp"
#include "PixFlow.hpp"
#include "PixFlow_GPU.hpp"

namespace optical_flow {
    using namespace std;
    using namespace cv;
    using namespace cv::detail;
    using namespace util;
    
    struct NovelViewUtil {
        static Vec4b generateNovelViewPoint(
                                            const Mat& srcImage,
                                            const Mat& flow,
                                            const double t,
                                            const int x,
                                            const int y);
        
        static Mat combineNovelViews(
                                     const Mat& imageL,
                                     const Mat& imageR,
                                     const Mat& flowLtoR,
                                     const Mat& flowRtoL,
                                     const Mat& blend);
    };
    
    class NovelViewGenerator {
    public:
        virtual ~NovelViewGenerator() {};
        
        virtual void prepare(
                             const Mat& colorImageL,
                             const Mat& colorImageR) = 0;
        
        virtual void generateNovelView(Mat& outNovelViewMerged) = 0;
        
        virtual Mat getFlowLtoR() { return Mat(); }
        virtual Mat getFlowRtoL() { return Mat(); }
        virtual void setBlend(const Mat& blend) = 0;
    };
    

    class NovelViewGeneratorAsymmetricFlow : public NovelViewGenerator {
    public:
        string flowAlgName;
        Mat imageL, imageR;
        Mat flowLtoR, flowRtoL;
        Mat Blend;
        
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
    
    Mat visualizeFlowAsGreyDisparity(const Mat& flow);
    
    Mat visualizeFlowAsVectorField(const Mat& flow, const Mat& image);
    
    Mat visualizeFlowColorWheel(const Mat& flow);
}

#endif /* OpticalFlow_hpp */
