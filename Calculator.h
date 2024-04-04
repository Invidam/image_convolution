//
// Created by Hansu Park on 2024/04/04.
//

#ifndef IMAGE_CONVOLUTION_CALCULATOR_H
#define IMAGE_CONVOLUTION_CALCULATOR_H
#include <omp.h>
#include <opencv2/opencv.hpp>

class Calculator {
public:
    static cv::Mat calculate(const cv::Mat &img);
};


#endif //IMAGE_CONVOLUTION_CALCULATOR_H
