//
//  util.cpp
//  OpticalFlow
//
//  Created by MungoMeng on 2017/7/24.
//  Copyright © 2017年 MungoMeng. All rights reserved.
//

#include "util.hpp"

#include <assert.h>
#include <execinfo.h>
#include <signal.h>

#include <fstream>
#include <map>
#include <string>
#include <random>
#include <exception>
#include <stdexcept>

namespace util {
    using namespace std;
    using namespace std::chrono;
    using namespace cv;

    Mat imreadExceptionOnFail(const string& filename, const int flags) {
        const Mat image = imread(filename, flags);
        if (image.empty()) {
            throw VrCamException("failed to load image: " + filename);
        }
        return image;
    }

    void imwriteExceptionOnFail(
                                const string& filename,
                                const Mat& image,
                                const vector<int>& params) {
        if (!imwrite(filename, image, params)) {
            throw VrCamException("failed to write image: " + filename);
        }
    }
    
    Mat stackHorizontal(const std::vector<Mat>& images) {
        assert(!images.empty());
        if (images.size() == 1) {
            return images[0];
        }
        Mat stacked = images[0].clone();
        for (int i = 1; i < images.size(); ++i) {
            hconcat(stacked, images[i], stacked);
        }
        return stacked;
    }
    
}
