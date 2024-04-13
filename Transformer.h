//
// Created by Hansu Park on 2024/04/04.
//

#ifndef IMAGE_CONVOLUTION_TRANSFORMER_H
#define IMAGE_CONVOLUTION_TRANSFORMER_H
#include <omp.h>
#include <opencv2/opencv.hpp>

class Transformer {
public:
    explicit Transformer(bool parallel = false);
    void parallel(bool parallel);
    cv::Mat convolve(const cv::Mat &img, const cv::Mat &filter) const;
private:
    bool _parallel;
    static int getPadding(int filter, int input, int stride);
    static cv::Mat broadcast(const cv::Mat &matrix, int channel_to);
};


#endif //IMAGE_CONVOLUTION_TRANSFORMER_H
