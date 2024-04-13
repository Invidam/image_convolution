//
// Created by Lazo Young on 13-Apr-24.
//

#ifndef IMAGE_CONVOLUTION_FILTER_H
#define IMAGE_CONVOLUTION_FILTER_H

#include "opencv2/core.hpp"

class Filter {
public:
    explicit Filter(const cv::Mat &image);
    Filter(const cv::Mat &image, bool parallel);
    void setParallelMode(bool parallel);
    cv::Mat gaussianBlur(int filter_size, float sigma) const;
    cv::Mat sobel(float sigma) const;
private:
    [[maybe_unused]] bool is_parallel;
    Transformer transformer;
    cv::Mat image;
};

#endif //IMAGE_CONVOLUTION_FILTER_H
