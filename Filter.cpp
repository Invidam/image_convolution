//
// Created by Lazo Young on 13-Apr-24.
//

#include <cmath>
#include "Filter.h"

Filter::Filter() {
    this->n_row = 3;
    this->n_col = 3;
    this->_parallel = false;
}

Filter::Filter(int size, bool parallel) {
    if (size % 2 == 0) {
        ++size;  // make sure it's an odd-number
    }

    this->n_row = size;
    this->n_col = size;
    this->_parallel = parallel;
}

cv::Mat Filter::gaussian(float sigma) const {
    const static double pi = acos(-1);
    const int origin_row = n_row / 2;
    const int origin_col = n_col / 2;
    const double ssq = 2 * sigma * sigma;
    cv::Mat mat(n_row, n_col, CV_8U);
    uchar *ptr = mat.ptr(0);

#pragma omp parallel for if(_parallel)
    for (int i = 0; i < n_row * n_col; ++i) {
        int row = i / n_col;
        int col = i % n_col;
        int y = row - origin_row;
        int x = col - origin_col;
        double index = - (x * x + y * y) / ssq;
        ptr[i] = cv::saturate_cast<uchar>(UINT8_MAX * exp(index) / (pi * ssq));
    }

    return mat;
}

void Filter::parallel(bool parallel) {
    this->_parallel = parallel;
}

void Filter::size(int size) {
    this->n_row = size;
    this->n_col = size;
}
