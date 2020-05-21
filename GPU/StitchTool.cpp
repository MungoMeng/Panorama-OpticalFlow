//
//  StitchTool.cpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include "StitchTool.hpp"

namespace stitch_tools{
    using namespace std;
    using namespace cv;

    void Stitchtools::prepare(
                              const Mat &colorImageL,
                              const Mat &colorImageR){
        
        //讲输入的图片储存到ImageL、ImageR
        ImageL = colorImageL.clone();
        ImageR = colorImageR.clone();
        
        //生成Map
        Stitchtools::MatchImages();
        
        //取出Map中值为150的部分，二值化为0与1后存入mapM，表示重叠区域的范围
        Mat mapM;
        threshold(Map, mapM , 140, 1, CV_THRESH_BINARY);
        //将ImageL分离成4个通道，分别与mapM按像素点乘法运算，留下重合区域存入OverlappedL
        vector<Mat> channels;
        split(ImageL,channels);
        channels.at(0) = channels.at(0).mul(mapM);
        channels.at(1) = channels.at(1).mul(mapM);
        channels.at(2) = channels.at(2).mul(mapM);
        channels.at(3) = channels.at(3).mul(mapM);
        merge(channels,OverlappedL);
        //将ImageR分离成4个通道，分别与mapM按像素点乘法运算，留下重合区域存入OverlappedR
        split(ImageR,channels);
        channels.at(0) = channels.at(0).mul(mapM);
        channels.at(1) = channels.at(1).mul(mapM);
        channels.at(2) = channels.at(2).mul(mapM);
        channels.at(3) = channels.at(3).mul(mapM);
        merge(channels,OverlappedR);
        
        //生成Blend
        //判断是否存在可用的GPU设备
        bool GPU = true;
        int num_devices = cuda::getCudaEnabledDeviceCount();
        if(num_devices <= 0){
            GPU = false;
            cout << "There is no GPU device." <<endl;
        }
        else
            cout << "There is "<<num_devices<< " GPU device." <<endl;

        int enable_device_id = -1;
        for(int i = 0; i < num_devices; i++){
            cuda::DeviceInfo dev_info(i);
            if(dev_info.isCompatible())   enable_device_id = i;
        }
        if(enable_device_id < 0){
            GPU = false;
            cout << "GPU module isn't built. Use CPU to generate Blend." <<endl;
        }
        else
            cout << "GPU "<<enable_device_id<<" module is built. Use GPU to generate Blend" <<endl;
        
        //若GPU为true，则选择GPU模式计算融合系数BLend
        if(GPU == true){
            //设置选用的GPU模块
            cuda::setDevice(enable_device_id);
            Stitchtools::GenerateBlend_GPU();
        }
        //若GPU为false，则选择普通模式计算融合系数Blend
        else    Stitchtools::GenerateBlend();
    }
    
    void Stitchtools::MatchImages(){
        //将左右原图的各通道分离，后面步骤仅需要其alpha通道
        vector<Mat> Lchannels,Rchannels;
        split(ImageL,Lchannels);
        split(ImageR,Rchannels);
        
        //初始化Map为32深度1通道的全零值矩阵
        Map = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));
        //将左图中alpha通道有效区域值置为100，右图中alpha通道有效区域值置为50
        threshold(Lchannels.at(3), Lchannels.at(3), 0, 100, CV_THRESH_BINARY);
        threshold(Rchannels.at(3), Rchannels.at(3), 0, 50, CV_THRESH_BINARY);
        
        //生成Map。左原图：100;重合区域：150;右原图：50;
        Map = Lchannels.at(3) + Rchannels.at(3);
    }
    
    void Stitchtools::Gather(){
        //定义变量，并将result初始化为4通道零值矩阵
        Mat GrayM;
        Mat map;
        Mat result = Mat(ImageL.rows, ImageL.cols, CV_8UC4,Scalar(0));
        
        //讲融合部分图像通道分离
        vector<Mat> channels;
        split(Mergedmiddle,channels);
        //将alpha通道二值化，有效区域值置为75
        threshold(channels.at(3), channels.at(3), 0, 75, CV_THRESH_BINARY);
        //map储存加入融合图像部分后的图像区域分布
        //左原图:100；右原图：50；融合部分：225||125||175；重合区域但非融合部分的区域：150
        map = Map + channels.at(3);
        
        //遍历像素点，按照map的区域划分，从不同的地方取像素点组成result
        for (int y = 0; y < ImageL.rows; ++y) {
            for (int x = 0; x < ImageL.cols; ++x) {
                //如果map=100，取左原图
                if(map.at<uchar>(y,x) == 100)
                    result.at<Vec4b>(y,x) = ImageL.at<Vec4b>(y,x);
                //如果map=50，取右原图
                else if(map.at<uchar>(y,x) == 50)
                    result.at<Vec4b>(y,x) = ImageR.at<Vec4b>(y,x);
                //如果map=225||125||175，取融合区域图
                else if(map.at<uchar>(y,x) == 225 || map.at<uchar>(y,x) == 125 || map.at<uchar>(y,x) == 175)
                    result.at<Vec4b>(y,x) = Mergedmiddle.at<Vec4b>(y,x);
                //如果map=150，则比较其与左右原图区域的距离，从距离小的区域的对应点取值
                else if(map.at<uchar>(y,x) == 150){
                    //在8方向上探索，最大探索距离100像素，若遇到map=100||50的区域则进行相应的操作
                    for (int i = 1; i < 100; i++){
                        if(map.at<uchar>(y,x+i) == 100 || map.at<uchar>(y,x-i) == 100 || map.at<uchar>(y+i,x) == 100 || map.at<uchar>(y-i,x) == 100 || map.at<uchar>(y-i,x-i) == 100 || map.at<uchar>(y-i,x+i) == 100 || map.at<uchar>(y+i,x-i) == 100 || map.at<uchar>(y+i,x+i) == 100){
                            //从左原图对应区域取点，然后退出循环
                            result.at<Vec4b>(y,x) = ImageL.at<Vec4b>(y,x);
                            break;
                        }
                        else if(map.at<uchar>(y,x+i) == 50 || map.at<uchar>(y,x-i) == 50 || map.at<uchar>(y+i,x) == 50 || map.at<uchar>(y-i,x) == 50 || map.at<uchar>(y-i,x-i) == 50 || map.at<uchar>(y-i,x+i) == 50 || map.at<uchar>(y+i,x-i) == 50 || map.at<uchar>(y+i,x+i) == 50){
                            //从右原图对应区域取点，然后退出循环
                            result.at<Vec4b>(y,x) = ImageR.at<Vec4b>(y,x);
                            break;
                        }
                        else
                            result.at<Vec4b>(y,x) = Vec4b(0,0,0,255);
                    }
                }
                //对map=0的无效区域，result的4通道置为0
                else if(map.at<uchar>(y,x) == 0)
                    result.at<Vec4b>(y,x) = Vec4b(0,0,0,0);
            }
        }
        FinalResult = result.clone();
    }
    
    void Stitchtools::GenerateBlend(){
        //定义变量并将其初始化为零值矩阵
        Mat blend = Mat(ImageL.rows, ImageL.cols, CV_32FC1,Scalar(0));
        
        //考虑全景图片左边界与右边界的连续性，将Map延扩后进行Blend计算
        //定义延扩长度
        int length = Map.cols/5;
        //定义一个向右平移length的平移矩阵
        Mat shftMat = (Mat_<double>(3,3)<<1,0,length, 0,1,0, 0,0,1);
        //对Map图进行延拓
        Mat Lpart,Rpart;
        Lpart = Map(Range(0,Map.rows),Range(0,length));
        Rpart = Map(Range(0,Map.rows),Range(Map.cols-length,Map.cols));
        warpPerspective(Map,Map,shftMat,Size(Map.cols+2*length,Map.rows),INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0,0));
        Rpart.copyTo(Mat(Map,Rect(0,0,length,Map.rows)));
        Lpart.copyTo(Mat(Map,Rect(Map.cols-length,0,length,Map.rows)));
        
        //预定义MergedDis，在计算Blend过程中计算得到
        MergedDis = Mat(Map.rows, Map.cols, CV_32FC1,Scalar(0));
        
        //遍历所有像素点，逐点计算融合系数blend
        for (int y = 0; y < ImageL.rows; ++y) {
            for (int x = 0; x < ImageL.cols; ++x) {
                //若为左原图的区域，blend = 0
                if(Map.at<uchar>(y,x+length) == 100)      blend.at<float>(y,x) = 0;
                //若为右原图的区域，blend = 1
                else if(Map.at<uchar>(y,x+length) == 50)  blend.at<float>(y,x) = 1;
                //若为重合区域，调用countblend函数计算
                else if(Map.at<uchar>(y,x+length) == 150) blend.at<float>(y,x) = countblend(x+length, y);
                //若为空白区域，blend = 0.5
                else                               blend.at<float>(y,x) = 0.5;
            }
        }
        
        //对延扩后生成的Map与MergedDis进行裁剪
        Map = Map(Range(0,Map.rows),Range(length,Map.cols-length)).clone();
        MergedDis = MergedDis(Range(0,MergedDis.rows),Range(length,MergedDis.cols-length)).clone();
        
        //计算step，为循环步长
        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;
        //对step为边长的小正方块进行平滑滤波操作
        //此操作希望仅对重合区域内部进行平滑操作，减小对于重合区域边界的影响
        for (int y = 0; y+step < ImageL.rows; y=y+step) {
            for (int x = 0; x+step < ImageL.cols; x=x+step) {
                //仅对两图重合区域中极力重合区域边界距离大于step的小正方块进行操作
                if(MergedDis.at<float>(y,x) > step){
                    Mat temp = blend(Range(y,y+step),Range(x,x+step));
                    blur(temp,temp,Size(ImageL.rows/130,ImageL.rows/130));
                }
            }
        }
        //对其余部分进行平滑滤波
        blur(blend,blend,Size(ImageL.rows/400,ImageL.rows/400));
        
        //结果存在Blend中
        Blend = blend.clone();
    }
    
    float Stitchtools::countblend(const int x, const int y){
        //该函数通过对(x,y)点八个方向的探索，得到该点与重合区域左边界和右边界的近似最小距离，通过最小距离进一步计算融合系数blend
        //与左原图接近的点blend值就趋近于0，而与右原图接近的点blend值趋近于1
        float blend;
        
        //计算循环步长step
        int step;
        if(ImageL.cols <= ImageL.rows) step = ImageL.cols/200; else step = ImageL.rows/200;
        
        //初始化minLdis，minRdis的值为极大值
        float minLdis = 10*ImageL.cols,minRdis = 10*ImageL.cols;
        
        //循环进行，对该点8个方向进行探索
        //记录与该点最近的左（右）原图边界到该点的距离
        for(int i = 0; i < ImageL.cols/2; i = i + step){        
            //对8个方向进行检索，若符合条件且minLdis（minRdis）值更小时，替换minLdis（minRdis）值
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
        
        //计算blend
        blend = minLdis/(minRdis+minLdis);
        //计算该点到重合区域边界的距离
        if(minLdis<minRdis) MergedDis.at<float>(y,x)=minLdis;
        else                MergedDis.at<float>(y,x)=minRdis;
        
        return blend;
    }
}
