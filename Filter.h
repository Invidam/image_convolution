//
// Created by Lazo Young on 13-Apr-24.
//

#ifndef IMAGE_CONVOLUTION_FILTER_H
#define IMAGE_CONVOLUTION_FILTER_H

#include "opencv2/core.hpp"

class Filter {
public:
    Filter();
    Filter(int size, bool parallel);
    void size(int size);
    void parallel(bool parallel);
    cv::Mat gaussian(float sigma) const;
private:
    int n_row;
    int n_col;
    bool _parallel;
};

#endif //IMAGE_CONVOLUTION_FILTER_H
