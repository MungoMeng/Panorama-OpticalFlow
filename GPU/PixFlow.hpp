//
//  PixFlow.hpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#ifndef PixFlow_h
#define PixFlow_h

#include <algorithm>
#include <cmath>

#include "util.hpp"

namespace optical_flow {
    using namespace std;
    using namespace cv;
    using namespace cv::detail;
    using namespace util;
    
    //用于计算两图间的光流偏移量，为基类，定义虚函数
    class OpticalFlowInterface {
    public:
        virtual ~OpticalFlowInterface() {};
        
        //标记图片所处相对位置
        enum class DirectionHint { UNKNOWN, RIGHT, DOWN, LEFT, UP };
        
        //计算I0BGRA到I1BGRA的光流偏移量
        virtual void computeOpticalFlow(
                                        const Mat& I0BGRA,
                                        const Mat& I1BGRA,
                                        Mat& flow,
                                        DirectionHint hint) = 0;
    };
    
    //用于计算两图间的光流偏移量
    template <
    int MaxPercentage = 0     // how far to look when initializing flow
    >
    struct PixFlow : public OpticalFlowInterface {
        //定义后续操作中需要用到的参数
        static constexpr int kPyrMinImageSize               = 24;
        static constexpr int kPyrMaxLevels                  = 1000;
        static constexpr float kGradEpsilon                 = 0.001f;  // for finite differences
        static constexpr float kUpdateAlphaThreshold        = 0.9f;    // pixels with alpha below this aren't updated by proposals
        static constexpr int kMedianBlurSize                = 5;       // medianBlur max size is 5 pixels for CV_32FC2
        static constexpr int kPreBlurKernelWidth            = 5;
        static constexpr float kPreBlurSigma                = 0.25f;   // amount to blur images before pyramids
        static constexpr int kFinalFlowBlurKernelWidth      = 3;
        static constexpr float kFinalFlowBlurSigma          = 1.0f;    // blur that is applied to flow at the end after upscaling
        static constexpr int kGradientBlurKernelWidth       = 3;
        static constexpr float kGradientBlurSigma           = 0.5f;    // amount to blur image gradients
        static constexpr int kBlurredFlowKernelWidth        = 15;      // for regularization/smoothing/diffusion
        static constexpr float kBlurredFlowSigma            = 8.0f;
        
        //随PixFlow模式变化而变换的参数
        //由PixFlow初始化时给入
        const float pyrScaleFactor;
        const float smoothnessCoef;
        const float verticalRegularizationCoef;
        const float horizontalRegularizationCoef;
        const float gradientStepSize;
        const float downscaleFactor;
        const float directionalRegularizationCoef;
        
        //构造函数
        PixFlow(
                const float pyrScaleFactor,
                const float smoothnessCoef,
                const float verticalRegularizationCoef,
                const float horizontalRegularizationCoef,
                const float gradientStepSize,
                const float downscaleFactor,
                const float directionalRegularizationCoef) :
        pyrScaleFactor(pyrScaleFactor),
        smoothnessCoef(smoothnessCoef),
        verticalRegularizationCoef(verticalRegularizationCoef),
        horizontalRegularizationCoef(horizontalRegularizationCoef),
        gradientStepSize(gradientStepSize),
        downscaleFactor(downscaleFactor),
        directionalRegularizationCoef(directionalRegularizationCoef) {}
        
        ~PixFlow() {}
        
        //下面函数均用于计算光流场
        void computeOpticalFlow(
                                const Mat& rgba0byte,
                                const Mat& rgba1byte,
                                Mat& flow,
                                DirectionHint hint) {
            
            //将图像降采样到更小的尺寸，提高处理速度
            Mat rgba0byteDownscaled, rgba1byteDownscaled;
            cv::Size originalSize = rgba0byte.size();
            cv::Size downscaleSize(
                                   rgba0byte.cols * downscaleFactor, rgba0byte.rows * downscaleFactor);
            resize(rgba0byte, rgba0byteDownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
            resize(rgba1byte, rgba1byteDownscaled, downscaleSize, 0, 0, CV_INTER_CUBIC);
            Mat motion = Mat(downscaleSize, CV_32F);
            
            //将图片变换到多种色度空间
            Mat I0Grey, I1Grey, I0, I1, alpha0, alpha1;
            vector<Mat> channels0, channels1;
            split(rgba0byteDownscaled, channels0);
            split(rgba1byteDownscaled, channels1);
            cvtColor(rgba0byteDownscaled, I0Grey,  CV_BGRA2GRAY);
            cvtColor(rgba1byteDownscaled, I1Grey,  CV_BGRA2GRAY);
            
            I0Grey.convertTo(I0, CV_32F);
            I1Grey.convertTo(I1, CV_32F);
            I0 /= 255.0f;
            I1 /= 255.0f;
            channels0[3].convertTo(alpha0, CV_32F);
            channels1[3].convertTo(alpha1, CV_32F);
            alpha0 /= 255.0f;
            alpha1 /= 255.0f;
            
            GaussianBlur(I0, I0, Size(kPreBlurKernelWidth, kPreBlurKernelWidth), kPreBlurSigma);
            GaussianBlur(I1, I1, Size(kPreBlurKernelWidth, kPreBlurKernelWidth), kPreBlurSigma);
            
            //建立图像金字塔
            vector<Mat> pyramidI0       = buildPyramid(I0);
            vector<Mat> pyramidI1       = buildPyramid(I1);
            vector<Mat> pyramidAlpha0   = buildPyramid(alpha0);
            vector<Mat> pyramidAlpha1   = buildPyramid(alpha1);
            vector<Mat> motionPyramid   = buildPyramid(motion);
            
            flow = Mat();
            for (int level = int(pyramidI0.size()) - 1; level >= 0; --level) {
                patchMatchPropagationAndSearch(
                                               pyramidI0[level],
                                               pyramidI1[level],
                                               pyramidAlpha0[level],
                                               pyramidAlpha1[level],
                                               flow,
                                               hint);
                
                if (level > 0) { // scale the flow up to the next size
                    resize(flow, flow, pyramidI0[level - 1].size(), 0, 0, CV_INTER_CUBIC);
                    flow *= (1.0f / pyrScaleFactor);
                }
            }
            
            //将所得结果变换为原尺寸图像
            resize(flow, flow, originalSize, 0, 0, CV_INTER_LINEAR);
            flow *= (1.0f / downscaleFactor);
            GaussianBlur(
                         flow,
                         flow,
                         Size(kFinalFlowBlurKernelWidth, kFinalFlowBlurKernelWidth),
                         kFinalFlowBlurSigma);
        }
        
        vector<Mat> buildPyramid(const Mat& src) {
            vector<Mat> pyramid = {src};
            while (pyramid.size() < kPyrMaxLevels) {
                Size newSize(
                             pyramid.back().cols * pyrScaleFactor + 0.5f,
                             pyramid.back().rows * pyrScaleFactor + 0.5f);
                if (newSize.height <= kPyrMinImageSize || newSize.width <= kPyrMinImageSize) {
                    break;
                }
                Mat scaledImage;
                resize(pyramid.back(), scaledImage, newSize, 0, 0, CV_INTER_LINEAR);
                pyramid.push_back(scaledImage);
            }
            return pyramid;
        }
        
        static inline int computeSearchDistance() {
            return (kPyrMinImageSize * MaxPercentage + 50) / 100;
        }
        
        float computePatchError(
                                const Mat& i0, const Mat& alpha0, int i0x, int i0y,
                                const Mat& i1, const Mat& alpha1, int i1x, int i1y){
            //在5*5的方格中计算绝对距离之和
            static const int kPatchRadius = 2;
            float sad = 0;
            float alpha = 0;
            for (int dy = -kPatchRadius; dy <= kPatchRadius; ++dy) {
                const int d0y = i0y + dy;
                if (0 <= d0y && d0y < i0.rows) {
                    const int d1y = clamp(i1y + dy, 0, i1.rows - 1);
                    for (int dx = -kPatchRadius; dx <= kPatchRadius; ++dx) {
                        const int d0x = i0x + dx;
                        if (0 <= d0x && d0x < i0.cols) {
                            const int d1x = clamp(i1x + dx, 0, i1.cols - 1);
                            const float difference =
                            i0.at<float>(d0y, d0x) -
                            i1.at<float>(d1y, d1x);
                            sad += std::abs(difference);
                            alpha +=
                            alpha0.at<float>(d0y, d0x) *
                            alpha1.at<float>(d1y, d1x);
                        }
                    }
                }
            }
            // normalize sad to sum of alphas (fine to go to infinity)
            sad /= alpha;
            // scale sad as flow vector length increases to favor short vectors
            const float length = norm(Point2f(i1x - i0x, i1y - i0y));
            sad *= 1 + length / computeSearchDistance();
            return sad;
        }
        
        float computeIntensityRatio(
                                    const Mat& lhs, const Mat& lhsAlpha,
                                    const Mat& rhs, const Mat& rhsAlpha){
            // just scale by the ratio between the sums, attenuated by alpha
            float sumLhs = 0;
            float sumRhs = 0;
            for (int y = 0; y < lhs.rows; ++y) {
                for (int x = 0; x < lhs.cols; ++x) {
                    float alpha = lhsAlpha.at<float>(y, x) * rhsAlpha.at<float>(y, x);
                    sumLhs += alpha * lhs.at<float>(y, x);
                    sumRhs += alpha * rhs.at<float>(y, x);
                }
            }
            return sumLhs / sumRhs;
        }
        
        Rect computeSearchBox(DirectionHint hint) {
            // we search a rectangle that is a fraction of the coarsest pyramid level
            const int dist = computeSearchDistance();
            // the rectangle extends ortho to each side of the search direction
            static const int kRatio = 8; // aspect ratio of search box
            const int ortho = (dist + kRatio / 2) / kRatio;
            const int thickness = 2 * ortho + 1;
            switch (hint) {
                // opencv rectangles are left, top, width, height
                case DirectionHint::RIGHT: return Rect(0, -ortho, dist + 1, thickness);
                case DirectionHint::DOWN: return Rect(-ortho, 0, thickness, dist + 1);
                case DirectionHint::LEFT: return Rect(-dist, -ortho, dist + 1, thickness);
                case DirectionHint::UP: return Rect(-ortho, -dist, thickness, dist + 1);
                case DirectionHint::UNKNOWN: break; // silence warning
            }
            return Rect();
        }
        
        void adjustInitialFlow(
                               const Mat& I0,
                               const Mat& I1,
                               const Mat& alpha0,
                               const Mat& alpha1,
                               Mat& flow,
                               const DirectionHint hint) {
            // compute a version of I1 that matches I0's intensity
            // this is basically poor man's color correction
            Mat I1eq = I1 * computeIntensityRatio(I0, alpha0, I1, alpha1);
            
            // estimate the flow of each pixel in I0 by searching a rectangle
            Rect box = computeSearchBox(hint);
            for (int i0y = 0; i0y < I0.rows; ++i0y) {
                for (int i0x = 0; i0x < I0.cols; ++i0x) {
                    if (alpha0.at<float>(i0y, i0x) > kUpdateAlphaThreshold) {
                        // create affinity for (0,0) by using fraction of the actual error
                        float kFraction = 0.8f; // lower the fraction to increase affinity
                        float errorBest = kFraction * computePatchError(
                                                                        I0, alpha0, i0x, i0y,
                                                                        I1eq, alpha1, i0x, i0y);
                        int i1xBest = i0x, i1yBest = i0y;
                        // look for better patch in the box
                        for (int dy = box.y; dy < box.y + box.height; ++dy) {
                            for (int dx = box.x; dx < box.x + box.width; ++dx) {
                                int i1x = i0x + dx;
                                int i1y = i0y + dy;
                                if (0 <= i1x && i1x < I1.cols && 0 <= i1y && i1y < I1.rows) {
                                    float error = computePatchError(
                                                                    I0, alpha0, i0x, i0y,
                                                                    I1eq, alpha1, i1x, i1y);
                                    if (errorBest > error) {
                                        errorBest = error;
                                        i1xBest = i1x;
                                        i1yBest = i1y;
                                    }
                                }
                            }
                        }
                        // use the best match
                        flow.at<Point2f>(i0y, i0x) = Point2f(i1xBest - i0x, i1yBest - i0y);
                    }
                }
            }
        }
        
        void patchMatchPropagationAndSearch(
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
                    adjustInitialFlow(I0, I1, alpha0, alpha1, flow, hint);
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
            // sweep from top/left
            for (int y = 0; y < imgSize.height; ++y) {
                for (int x = 0; x < imgSize.width; ++x) {
                    if (alpha0.at<float>(y, x) > kUpdateAlphaThreshold && alpha1.at<float>(y, x) > kUpdateAlphaThreshold) {
                        float currErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x));
                        if (x > 0) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y, x - 1)); }
                        if (y > 0) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y - 1, x)); }
                        flow.at<Point2f>(y, x) -= gradientStepSize * errorGradient(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, currErr);
                    }
                }
            }
            medianBlur(flow, flow, kMedianBlurSize);
            
            // sweep from bottom/right
            for (int y = imgSize.height - 1; y >= 0; --y) {
                for (int x = imgSize.width - 1; x >= 0; --x) {
                    if (alpha0.at<float>(y, x) > kUpdateAlphaThreshold && alpha1.at<float>(y, x) > kUpdateAlphaThreshold) {
                        float currErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x));
                        if (x < imgSize.width - 1)  { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y, x + 1)); }
                        if (y < imgSize.height - 1) { proposeFlowUpdate(alpha0, alpha1, I0, I1, I0x, I0y, I1x, I1y, flow, blurredFlow, currErr, x, y, flow.at<Point2f>(y + 1, x)); }
                        flow.at<Point2f>(y, x) -= gradientStepSize * errorGradient(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y,x, y, flow, blurredFlow, currErr);
                    }
                }
            }
            medianBlur(flow, flow, kMedianBlurSize);
            lowAlphaFlowDiffusion(alpha0, alpha1, flow);
        }
        
        inline void proposeFlowUpdate(
                                      const Mat& alpha0,
                                      const Mat& alpha1,
                                      const Mat& I0,
                                      const Mat& I1,
                                      const Mat& I0x,
                                      const Mat& I0y,
                                      const Mat& I1x,
                                      const Mat& I1y,
                                      Mat& flow,
                                      const Mat& blurredFlow,
                                      float& currErr,
                                      const int updateX, const int updateY,
                                      const Point2f& proposedFlow) {
            
            const float proposalErr = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, updateX, updateY, flow, blurredFlow, proposedFlow);
            if (proposalErr < currErr) {
                flow.at<Point2f>(updateY, updateX) = proposedFlow;
                currErr = proposalErr;
            }
        }
        
        inline Point2f errorGradient(
                                     const Mat& I0,
                                     const Mat& I1,
                                     const Mat& alpha0,
                                     const Mat& alpha1,
                                     const Mat& I0x,
                                     const Mat& I0y,
                                     const Mat& I1x,
                                     const Mat& I1y,
                                     const int x,
                                     const int y,
                                     const Mat& flow,
                                     const Mat& blurredFlow,
                                     const float currErr) {
            
            const static Point2f dx(kGradEpsilon, 0.0f);
            const static Point2f dy(0.0f, kGradEpsilon);
            
            const float fx = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x) + dx);
            const float fy = errorFunction(I0, I1, alpha0, alpha1, I0x, I0y, I1x, I1y, x, y, flow, blurredFlow, flow.at<Point2f>(y, x) + dy);
            
            return Point2f((fx - currErr) / kGradEpsilon, (fy - currErr) / kGradEpsilon);
        }
        
        void lowAlphaFlowDiffusion(const Mat& alpha0, const Mat& alpha1, Mat& flow) {
            Mat blurredFlow;
            GaussianBlur(
                         flow,
                         blurredFlow,
                         Size(kBlurredFlowKernelWidth, kBlurredFlowKernelWidth),
                         kBlurredFlowSigma);
            for (int y = 0; y < flow.rows; ++y) {
                for (int x = 0; x < flow.cols; ++x) {
                    const float a0 = alpha0.at<float>(y, x);
                    const float a1 = alpha1.at<float>(y, x);
                    const float diffusionCoef = 1.0f - a0 * a1;
                    flow.at<Point2f>(y, x) =
                    diffusionCoef * blurredFlow.at<Point2f>(y, x)
                    + (1.0f - diffusionCoef) * flow.at<Point2f>(y, x);
                }
            }
        }
        
        static inline float getPixBilinear32FExtend(const Mat& img, float x, float y) {
            const cv::Size& imgSize = img.size();
            x                 = min(imgSize.width - 2.0f, max(0.0f, x));
            y                 = min(imgSize.height - 2.0f, max(0.0f, y));
            const int x0      = int(x);
            const int y0      = int(y);
            const float xR    = x - float(x0);
            const float yR    = y - float(y0);
            const float* p    = img.ptr<float>(y0);
            const float f00   = *(p + x0);
            const float f01   = *(p + x0 + img.cols);
            const float f10   = *(p + x0 + 1);
            const float f11   = *(p + x0 + img.cols + 1);
            const float a1    = f00;
            const float a2    = f10 - f00;
            const float a3    = f01 - f00;
            const float a4    = f00 + f11 - f10 - f01;
            return a1 + a2 * xR + a3 * yR + a4 * xR * yR;
        }
        
        inline float errorFunction(
                                   const Mat& I0,
                                   const Mat& I1,
                                   const Mat& alpha0,
                                   const Mat& alpha1,
                                   const Mat& I0x,
                                   const Mat& I0y,
                                   const Mat& I1x,
                                   const Mat& I1y,
                                   const int x,
                                   const int y,
                                   const Mat& flow,
                                   const Mat& blurredFlow,
                                   const Point2f& flowDir) {
            const float matchX      = x + flowDir.x;
            const float matchY      = y + flowDir.y;
            const float i0x         = I0x.at<float>(y, x);
            const float i0y         = I0y.at<float>(y, x);
            const float i1x         = getPixBilinear32FExtend(I1x, matchX, matchY);
            const float i1y         = getPixBilinear32FExtend(I1y, matchX, matchY);
            const Point2f flowDiff  = blurredFlow.at<Point2f>(y, x) - flowDir;
            const float smoothness  = sqrtf(flowDiff.dot(flowDiff));
            float err = sqrtf((i0x - i1x) * (i0x - i1x) + (i0y - i1y) * (i0y - i1y))
            + smoothness * smoothnessCoef
            + verticalRegularizationCoef * fabsf(flowDir.y) / float(I0.cols)
            + horizontalRegularizationCoef * fabsf(flowDir.x) / float(I0.cols);
            
            return err;
        }
    };
    
    //用于初始化模板类结构体PixFlow
    //输入不同的flowAlgName，对PixFlow进行不同模式的初始化
    static OpticalFlowInterface* makeOpticalFlowByName(const string flowAlgName) {
        //若选择pixflow_low模式，则进行以下操作
        if (flowAlgName == "pixflow_low") {
            static const float kPyrScaleFactor                  = 0.9f;
            static const float kSmoothnessCoef                  = 0.001f;
            static const float kVerticalRegularizationCoef      = 0.01f;
            static const float kHorizontalRegularizationCoef    = 0.01f;
            static const float kGradientStepSize                = 0.5f;
            static const float kDownscaleFactor                 = 0.5f;
            static const float kDirectionalRegularizationCoef   = 0.0f;
            return new PixFlow<0>(
                                  kPyrScaleFactor,
                                  kSmoothnessCoef,
                                  kVerticalRegularizationCoef,
                                  kHorizontalRegularizationCoef,
                                  kGradientStepSize,
                                  kDownscaleFactor,
                                  kDirectionalRegularizationCoef
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
            return new PixFlow<20>(
                                   kPyrScaleFactor,
                                   kSmoothnessCoef,
                                   kVerticalRegularizationCoef,
                                   kHorizontalRegularizationCoef,
                                   kGradientStepSize,
                                   kDownscaleFactor,
                                   kDirectionalRegularizationCoef
                                   );
        }
        //若flowAlgName为不可识别的字符串，则抛出错误信息
        throw VrCamException("unrecognized flow algorithm name: " + flowAlgName);
    }
}

#endif /* PixFlow_h */
