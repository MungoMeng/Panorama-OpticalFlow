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
    
    void printStacktrace() {
        const size_t maxStackDepth = 128;
        void* stack[maxStackDepth];
        size_t stackDepth = backtrace(stack, maxStackDepth);
        char** stackStrings = backtrace_symbols(stack, int(stackDepth));
        for (size_t i = 0; i < stackDepth; ++i) {
            LOG(ERROR) << stackStrings[i];
        }
        free(stackStrings);
    }
    
    void terminateHandler() {
        exception_ptr exptr = current_exception();
        if (exptr != 0) {
            try {
                rethrow_exception(exptr);
            } catch (VrCamException &ex) {
                LOG(ERROR) << "Terminated with VrCamException: " << ex.what();
            } catch (exception &ex) {
                LOG(ERROR) << "Terminated with exception: " << ex.what();
                printStacktrace();
            } catch (...) {
                LOG(ERROR) << "Terminated with unknown exception";
                printStacktrace();
            }
        } else {
            LOG(ERROR) << "Terminated due to unknown reason";
            printStacktrace();
        }
        abort();
    }
    
    void sigHandler(int signal) {
        LOG(ERROR) << strsignal(signal);
        printStacktrace();
        abort();
    }
    
    void initOpticalFlow(int argc, char** argv) {
        // Initialize Google's logging library
        google::InitGoogleLogging(argv[0]);
        // GFlags
        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        fLB::FLAGS_helpshort = fLB::FLAGS_help;
        fLB::FLAGS_help = false;
        gflags::HandleCommandLineHelpFlags();
        // setup signal and termination handlers
        set_terminate(terminateHandler);
        // terminate process: terminal line hangup
        signal(SIGHUP, sigHandler);
        // terminate process: interrupt program
        signal(SIGINT, sigHandler);
        // create core image: quit program
        signal(SIGQUIT, sigHandler);
        // create core image: illegal instruction
        signal(SIGILL, sigHandler);
        // create core image: trace trap
        signal(SIGTRAP, sigHandler);
        // create core image: floating-point exception
        signal(SIGFPE, sigHandler);
        // terminate process: kill program
        signal(SIGKILL, sigHandler);
        // create core image: bus error
        signal(SIGBUS, sigHandler);
        // create core image: segmentation violation
        signal(SIGSEGV, sigHandler);
        // create core image: non-existent system call invoked
        signal(SIGSYS, sigHandler);
        // terminate process: write on a pipe with no reader
        signal(SIGPIPE, sigHandler);
        // terminate process: software termination signal
        signal(SIGTERM, sigHandler);
    }
}
