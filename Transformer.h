//
// Created by Hansu Park on 2024/04/04.
//

#ifndef IMAGE_CONVOLUTION_TRANSFORMER_H
#define IMAGE_CONVOLUTION_TRANSFORMER_H
#include <omp.h>
#include <opencv2/opencv.hpp>

class Transformer {
public:
    explicit Transformer(bool parallel = false, bool verbose = false);
    void setParallelMode(bool mode);
    cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel) const;
    cv::Mat convolve(const cv::Mat &image, const std::vector<cv::Mat> &kernels, const std::function<int(int[])> &reduce) const;
private:
    [[maybe_unused]] bool parallel;
    bool verbose;
    static int getPadding(int filter, int input, int stride);
    static cv::Mat broadcast(const cv::Mat &src, int n_channel);
    static void permute(cv::Mat &src, int n_channel, int type) ;
};


#endif //IMAGE_CONVOLUTION_TRANSFORMER_H
