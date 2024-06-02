//
// Created by Lazo Young on 13-Apr-24.
//

#ifndef IMAGE_CONVOLUTION_FILTER_H
#define IMAGE_CONVOLUTION_FILTER_H

#include "opencv2/core.hpp"
#include "Transformer.h"
#include "Timer.h"

class Filter {
public:
    explicit Filter(const cv::Mat &image);
    Filter(const cv::Mat &image, bool parallel, bool verbose = false);
    void setParallelMode(bool parallel);
    void setCudaMode(bool isCuda);
    cv::Mat gaussianBlur(int filter_size, float sigma) const;
    cv::Mat sobel(float sigma) const;
    cv::Mat opencvGaussianBlur(float sigma, int filter_size) const;
    cv::Mat opencvSobel(int ddepth, int dx, int dy, int ksize) const;
private:
    bool is_parallel;
    bool is_cuda;
    bool verbose;
    Transformer transformer;
    cv::Mat image;

    void
    foo(const double pi, const int n_row, const int n_col, const int origin_row, const int origin_col, const double ssq,
        int *ptr) const;
};

#endif //IMAGE_CONVOLUTION_FILTER_H
