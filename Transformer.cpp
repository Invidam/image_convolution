//
// Created by Hansu Park on 2024/04/04.
//

#include "Transformer.h"

Transformer::Transformer(bool parallel) {
    this->_parallel = parallel;
}

cv::Mat Transformer::convolve(const cv::Mat &img, const cv::Mat &filter) const { // NOLINT(*-convert-member-functions-to-static)
    const int Fh = filter.rows;
    const int Fw = filter.cols;
    const int Oh = img.rows;
    const int Ow = img.cols;
    const int Ph = getPadding(Fh, Oh, 1);
    const int Pw = getPadding(Fw, Ow, 1);
    const int CH = img.channels();
    const int Ih = Oh + 2 * Ph;
    const int Iw = Ow + 2 * Pw;

    CV_Assert(img.isContinuous());
    CV_Assert(filter.isContinuous());
    CV_Assert(Fw <= Ow);
    CV_Assert(Fh <= Oh);     // filter size never surpass image
    CV_Assert(Fh % 2 == 1);  // filter size is always odd-number
    CV_Assert(Fh == Fw);     // filter shape is always square

    cv::Mat output(img);
    cv::Mat _filter(filter);
    cv::Mat _img(Ih, Iw, img.type());
    cv::copyMakeBorder(img, _img, Ph, Ph, Pw, Pw, cv::BORDER_ISOLATED);

    // Make sure img and filter have same number of channels.
    if (_filter.channels() != CH) {
        _filter = broadcast(_filter.clone(), CH);
        CV_Assert(_filter.isContinuous());
    }

    uchar *o_ptr = output.ptr(0);
    uint divisor = static_cast<uint>(cv::sum(filter).val[0]);
    const uchar *i_ptr = _img.ptr(0);
    const uchar *f_ptr = _filter.ptr(0);
    const int v_step = Ih - Fh;
    const int h_step = Iw - Fw;

    if (divisor < 1)
        divisor = 1;

#pragma omp parallel for if(_parallel)
    for (int i = 0; i < v_step * h_step; ++i) {
        int row = i / h_step;
        int col = i % h_step;
        int start = (row * Iw + col) * CH;
        std::vector<uint> sum(CH, 0);

        // Compute the convolution of present point.
        // Point (row, col) is where the filter's top-left pixel overlaps.
        for (int j = 0; j < Fh * Fw; ++j) {
            const int r = j / Fw;
            const int c = j % Fw;
            const int i_idx = start + (r * Iw + c) * CH;
            const int f_idx = (r * Fw + c) * CH;

            for (int ch = 0; ch < CH; ++ch) {
                sum[ch] += i_ptr[i_idx + ch] * f_ptr[f_idx + ch];
            }
        }

        // Point (row+1, col+1) is where the center of filter overlaps.
        // Let's update this particular pixel for output.
        start = (row * Ow + col) * CH;
        int o_idx = start + (Ow + 1) * CH;

        for (int ch = 0; ch < CH; ++ch) {
            o_ptr[o_idx + ch] = sum[ch] / divisor;
        }
    }

    return output;
}

int Transformer::getPadding(int filter, int input, int stride) {
    // O = I = (I - F + 2P) / S + 1
    // P = ((I - 1) * S - I + F) / 2
    return ((input - 1) * stride - input + filter) / 2;
}

cv::Mat Transformer::broadcast(const cv::Mat &matrix, int channel_to) {
    cv::Mat dst;
    cv::Mat m = cv::Mat::ones(channel_to, 1, matrix.type());
    cv::transform(matrix, dst, m);
    return dst;
}

void Transformer::parallel(bool parallel) {
    this->_parallel = parallel;
}
