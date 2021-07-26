//g++ TWY-Sign-Detection.h -I/usr/include/opencv4 -I/usr/include/python3.9 -lopencv_core -lopencv_imgproc -lopencv_highgui

#ifndef TWY_SIGN_DETECTION_H
#define TWY_SIGN_DETECTION_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct SignText;

struct SignImages;

class TwySignDetector {

    public:
        SignText getSignText();
    
    private:
        cv::Mat detect_sign(cv::Mat I, int max_dim, int net_step, int out_size, int threshold);
        SignImages getSegments(cv::Mat img);
        double colourDistance(cv::Mat colour1, cv::Mat colour2);

};

#endif