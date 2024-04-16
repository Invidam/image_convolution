//
// Created by Lazo Young on 13-Apr-24.
//

#ifndef IMAGE_CONVOLUTION_FILTER_H
#define IMAGE_CONVOLUTION_FILTER_H

#include "opencv2/core.hpp"

class Filter {
public:
    explicit Filter(const cv::Mat &image);
    Filter(const cv::Mat &image, bool parallel, bool verbose = false);
    void setParallelMode(bool parallel);
    cv::Mat gaussianBlur(int filter_size, float sigma) const;
    cv::Mat sobel(float sigma) const;
    cv::Mat opencvGaussianBlur(float sigma, int filter_size) const;
    cv::Mat opencvSobel(int ddepth, int dx, int dy, int ksize) const;
private:
    bool is_parallel;
    bool verbose;
    Transformer transformer;
    cv::Mat image;
};

#endif //IMAGE_CONVOLUTION_FILTER_H
