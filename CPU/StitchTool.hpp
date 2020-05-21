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

#include <opencv2/opencv.hpp>

namespace stitch_tools{
    using namespace std;
    using namespace cv;
    
    class Stitchtools{
    public:
        Mat ImageL,ImageR;
        Mat Blend;
        Mat OverlappedL,OverlappedR;
        Mat Mergedmiddle;                  //Merged image in the area3
        Mat Map;                           //Area map on the canvas。ImageL(area1): 100; ImageR(area12): 50; Overlap(area3): 150
        Mat FinalResult;
        Mat MergedDis;                     
        
        Stitchtools() {}
        ~Stitchtools() {}
        
        //calculate OverlappedL、OverlappedR、Map、Blend
        void prepare(
                     const Mat& colorImageL,
                     const Mat& colorImageR);
        
        //generate canvas map
        void MatchImages();
        
        //generate final result from Mergedmiddle and original images
        void Gather();
        
        void GenerateBlend();
        
        float countblend(
                         const int x,
                         const int y);
        
        Mat getImageL() { return ImageL; }
        Mat getImageR() { return ImageR; }
        Mat getBlend()  { return Blend;  }
        Mat getMap()    { return Map;    }
        Mat getOverlappedL() { return OverlappedL; }
        Mat getOverlappedR() { return OverlappedR; }
        Mat getFinalResult() { return FinalResult; }
        
        void setMergedmiddle(const Mat &image) { Mergedmiddle = image.clone();  }
        
    };
}

#endif /* StitchTool_hpp */
