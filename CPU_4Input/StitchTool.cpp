#include "StitchTool.hpp"

namespace stitch_tools{
    using namespace std;
    using namespace cv;

    void Stitchtools::prepare(
                              const Mat &colorImageL,
                              const Mat &colorImageR){
        
        ImageL = colorImageL.clone();
        ImageR = colorImageR.clone();
        
        //generate canvas map
        Stitchtools::MatchImages();
        
        Mat mapM;
        threshold(Map, mapM , 140, 1, CV_THRESH_BINARY);

        vector<Mat> channels;
        split(ImageL,channels);
        channels.at(0) = channels.at(0).mul(mapM);
        channels.at(1) = channels.at(1).mul(mapM);
        channels.at(2) = channels.at(2).mul(mapM);
        channels.at(3) = channels.at(3).mul(mapM);
        merge(channels,OverlappedL);
        
        split(ImageR,channels);
        channels.at(0) = channels.at(0).mul(mapM);
        channels.at(1) = channels.at(1).mul(mapM);
        channels.at(2) = channels.at(2).mul(mapM);
        channels.at(3) = channels.at(3).mul(mapM);
        merge(channels,OverlappedR);
        
        Stitchtools::GenerateBlend();
    }
    
    void Stitchtools::MatchImages(){
        //Split channels, and only alpha channel is needed here.
        vector<Mat> Lchannels,Rchannels;
        split(ImageL,Lchannels);
        split(ImageR,Rchannels);
        
        Map = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));

        threshold(Lchannels.at(3), Lchannels.at(3), 0, 100, CV_THRESH_BINARY);
        threshold(Rchannels.at(3), Rchannels.at(3), 0, 50, CV_THRESH_BINARY);
        
        Map = Lchannels.at(3) + Rchannels.at(3);
    }
    
    void Stitchtools::Gather(){
        Mat GrayM;
        Mat map;
        Mat result = Mat(ImageL.rows, ImageL.cols, CV_8UC4,Scalar(0));
        
        vector<Mat> channels;
        split(Mergedmiddle,channels);
        threshold(channels.at(3), channels.at(3), 0, 75, CV_THRESH_BINARY);

        map = Map + channels.at(3);

        for (int y = 0; y < ImageL.rows; ++y) {
            for (int x = 0; x < ImageL.cols; ++x) {
                
                if(map.at<uchar>(y,x) == 100)
                    result.at<Vec4b>(y,x) = ImageL.at<Vec4b>(y,x);
                
                else if(map.at<uchar>(y,x) == 50)
                    result.at<Vec4b>(y,x) = ImageR.at<Vec4b>(y,x);
                
                else if(map.at<uchar>(y,x) == 225 || map.at<uchar>(y,x) == 125 || map.at<uchar>(y,x) == 175)
                    result.at<Vec4b>(y,x) = Mergedmiddle.at<Vec4b>(y,x);
                
                else if(map.at<uchar>(y,x) == 150){
                    //search in 8 directions
                    for (int i = 1; i < 100; i++){
                        if(map.at<uchar>(y,x+i) == 100 || map.at<uchar>(y,x-i) == 100 || map.at<uchar>(y+i,x) == 100 || map.at<uchar>(y-i,x) == 100 || map.at<uchar>(y-i,x-i) == 100 || map.at<uchar>(y-i,x+i) == 100 || map.at<uchar>(y+i,x-i) == 100 || map.at<uchar>(y+i,x+i) == 100){
                            result.at<Vec4b>(y,x) = ImageL.at<Vec4b>(y,x);
                            break;
                        }
                        else if(map.at<uchar>(y,x+i) == 50 || map.at<uchar>(y,x-i) == 50 || map.at<uchar>(y+i,x) == 50 || map.at<uchar>(y-i,x) == 50 || map.at<uchar>(y-i,x-i) == 50 || map.at<uchar>(y-i,x+i) == 50 || map.at<uchar>(y+i,x-i) == 50 || map.at<uchar>(y+i,x+i) == 50){
                            result.at<Vec4b>(y,x) = ImageR.at<Vec4b>(y,x);
                            break;
                        }
                        else
                            result.at<Vec4b>(y,x) = Vec4b(0,0,0,255);
                    }
                }
                
                else if(map.at<uchar>(y,x) == 0)
                    result.at<Vec4b>(y,x) = Vec4b(0,0,0,0);
            }
        }
        FinalResult = result.clone();
    }
    
    void Stitchtools::GenerateBlend(){
        Mat blend = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));
        
        //map and blend are extended to increase the continuity between lift and right boundary
        int length = Map.cols/5;
        
        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);
        //extend map
        Mat Lpart,Rpart;
        Lpart = Map(Range(0,Map.rows),Range(0,length));
        Rpart = Map(Range(0,Map.rows),Range(Map.cols-length,Map.cols));
        warpPerspective(Map,Map,shftMat,Size(Map.cols+2*length,Map.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(Map,Rect(0,0,length,Map.rows)));
        Lpart.copyTo(Mat(Map,Rect(Map.cols-length,0,length,Map.rows)));
        
        MergedDis = Mat(Map.rows, Map.cols, CV_32FC1,Scalar(0));
        for (int y = 0; y < ImageL.rows; ++y) {
            for (int x = 0; x < ImageL.cols; ++x) {
                
                if(Map.at<uchar>(y,x+length) == 100)     blend.at<float>(y,x) = 0;
                
                else if(Map.at<uchar>(y,x+length) == 50)  blend.at<float>(y,x) = 1;
                
                else if(Map.at<uchar>(y,x+length) == 150) blend.at<float>(y,x) = countblend(x+length, y);
                
                else                               blend.at<float>(y,x) = 0.5;
            }
        }
        
        Map = Map(Range(0,Map.rows),Range(length,Map.cols-length)).clone();
        MergedDis = MergedDis(Range(0,MergedDis.rows),Range(length,MergedDis.cols-length)).clone();
        
        //smoothen
        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;
        
        for (int y = 0; y+step < ImageL.rows; y=y+step) {
            for (int x = 0; x+step < ImageL.cols; x=x+step) {
                
                if(MergedDis.at<float>(y,x) > step){
                    Mat temp = blend(Range(y,y+step),Range(x,x+step));
                    blur(temp,temp,Size(ImageL.rows/130,ImageL.rows/130));
                }
            }
        }
        blur(blend,blend,Size(ImageL.rows/400,ImageL.rows/400));
        
        Blend = blend.clone();
    }
    
    float Stitchtools::countblend(const int x, const int y){
        float blend;
        
        //loop step
        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;
        
        float minLdis = 10*ImageL.cols,minRdis = 10*ImageL.cols;
        
        //search in 8 directions
        for(int i = 0; i < ImageL.cols/2; i = i + step){
            if(x+i < Map.cols && Map.at<uchar>(y,x+i) == 100 && i<minLdis)  minLdis = i;
            if(x+i < Map.cols && Map.at<uchar>(y,x+i) == 50  && i<minRdis)  minRdis = i;
            if(x-i > 0           && Map.at<uchar>(y,x-i) == 100 && i<minLdis)  minLdis = i;
            if(x-i > 0           && Map.at<uchar>(y,x-i) == 50  && i<minRdis)  minRdis = i;
            if(y+i < Map.rows && Map.at<uchar>(y+i,x) == 100 && i<minLdis)  minLdis = i;
            if(y+i < Map.rows && Map.at<uchar>(y+i,x) == 50  && i<minRdis)  minRdis = i;
            if(y-i > 0           && Map.at<uchar>(y-i,x) == 100 && i<minLdis)  minLdis = i;
            if(y-i > 0           && Map.at<uchar>(y-i,x) == 50  && i<minRdis)  minRdis = i;
            if((x+i < Map.cols && y+i < Map.rows) && Map.at<uchar>(y+i,x+i) == 100 && i*sqrt(2)<minLdis)
                minLdis = i*sqrt(2);
            if((x+i < Map.cols && y+i < Map.rows) && Map.at<uchar>(y+i,x+i) == 50  && i*sqrt(2)<minRdis)
                minRdis = i*sqrt(2);
            if((x-i > 0           && y-i > 0          ) && Map.at<uchar>(y-i,x-i) == 100 && i*sqrt(2)<minLdis)
                minLdis = i*sqrt(2);
            if((x-i > 0           && y-i > 0          ) && Map.at<uchar>(y-i,x-i) == 50  && i*sqrt(2)<minRdis)
                minRdis = i*sqrt(2);
            if((x+i < Map.cols && y-i > 0          ) && Map.at<uchar>(y-i,x+i) == 100 && i*sqrt(2)<minLdis)
                minLdis = i*sqrt(2);
            if((x+i < Map.cols && y-i > 0          ) && Map.at<uchar>(y-i,x+i) == 50  && i*sqrt(2)<minRdis)
                minRdis = i*sqrt(2);
            if((x-i > 0           && y+i < Map.rows) && Map.at<uchar>(y+i,x-i) == 100 && i*sqrt(2)<minLdis)
                minLdis = i*sqrt(2);
            if((x-i > 0           && y+i < Map.rows) && Map.at<uchar>(y+i,x-i) == 50  && i*sqrt(2)<minRdis)
                minRdis = i*sqrt(2);
        }
        
        blend = minLdis/(minRdis+minLdis);
        
        if(minLdis<minRdis) MergedDis.at<float>(y,x)=minLdis;
        else                MergedDis.at<float>(y,x)=minRdis;
        
        return blend;
    }
}
