#include "Transformer.h"
#include "Timer.h"

Transformer::Transformer(bool parallel, bool verbose) {
    this->parallel = parallel;
    this->verbose = verbose;
}

cv::Mat Transformer::convolve(  // NOLINT(*-convert-member-functions-to-static)
        const cv::Mat &image,
        const std::vector<cv::Mat> &kernels,
        const std::function<int(int[])> &reduce
) const {
    const int Fh = kernels[0].rows;
    const int Fw = kernels[0].cols;
    const int Oh = image.rows;
    const int Ow = image.cols;
    const int Ph = getPadding(Fh, Oh, 1);
    const int Pw = getPadding(Fw, Ow, 1);
    const int CH = image.channels();
    const int Ih = Oh + 2 * Ph;
    const int Iw = Ow + 2 * Pw;

    CV_Assert(image.isContinuous());
    CV_Assert(kernels[0].isContinuous());
    CV_Assert(Fw <= Ow);
    CV_Assert(Fh <= Oh);     // kernel size never surpass image
    CV_Assert(Fh % 2 == 1);  // kernel size is always odd-number
    CV_Assert(Fh == Fw);     // kernel shape is always square

    cv::Mat _image(Ih, Iw, image.type());
    cv::Mat output(image);

    cv::copyMakeBorder(image, _image, Ph, Ph, Pw, Pw, cv::BORDER_ISOLATED);
    permute(_image, CH, CV_8U);  // 8-bit unsigned char (uchar)

    const int v_step = Ih - Fh;
    const int h_step = Iw - Fw;
    const uchar *i_ptr = _image.ptr(0);
    uchar *o_ptr = output.ptr(0);
    const size_t K = kernels.size();
    std::vector<cv::Mat> _kernel(K);
    std::vector<int> divisor(K);
    std::vector<const int *> f_ptr(K);

    for (int i = 0; i < K; ++i) {
        cv::Mat kernel(kernels[i]);
        permute(kernel, CH, CV_32S);  // 32-bit signed integer
        _kernel[i] = kernel;
        divisor[i] = std::max(1, static_cast<int>(cv::sum(kernel).val[0]));
        f_ptr[i] = kernel.ptr<int>(0);
    }

#pragma omp parallel for if(parallel)
    for (int i = 0; i < v_step * h_step; ++i) {
        const int row = i / h_step;
        const int col = i % h_step;
        int start = (row * Iw + col) * CH;
        std::vector<int> sum(CH * K, 0);
        Timer timer;

        // Compute the convolution of present point.
        // Point (row, col) is where the kernel's top-left pixel overlaps.
        for (int j = 0; j < Fh * Fw; ++j) {
            const int r = j / Fw;
            const int c = j % Fw;
            const int i_idx = start + (r * Iw + c) * CH;
            const int f_idx = (r * Fw + c) * CH;

            for (int ch = 0; ch < CH; ++ch) {
                uchar color = i_ptr[i_idx + ch];

                for (int k = 0; k < K; ++k) {
                    sum[k * CH + ch] += color * f_ptr[k][f_idx + ch];
                }
            }
        }

        if (verbose) {
            auto time = timer.elapsed();
            std::cout << "Pixel (" << row << ',' << col << ") convolved in " << time << " ms" << std::endl;
            timer.reset();
        }

        // Point (row+1, col+1) is where the center of kernel overlaps.
        // Let's update this particular pixel for output.
        start = (row * Ow + col) * CH;
        int o_idx = start + (Ow + 1) * CH;

        for (int ch = 0; ch < CH; ++ch) {
            std::vector term(K, 0);

            for (int k = 0; k < K; ++k) {
                term[k] = std::abs(sum[ch] / divisor[k]);
            }

            int color = (K > 1) ? reduce(term.data()) : term[0];
            o_ptr[o_idx + ch] = cv::saturate_cast<uint>(color);
        }

        if (verbose) {
            auto time = timer.elapsed();
            std::cout << "Pixel (" << row + 1 << ',' << col + 1 << ") updated in " << time << " ms" << std::endl;
            timer.reset();
        }
    }

    return output;
}

cv::Mat Transformer::convolve(const cv::Mat &image,
                              const cv::Mat &kernel) const { // NOLINT(*-convert-member-functions-to-static)
    std::vector kernels(1, kernel);
    std::function<int(int[])> reduce;
    return convolve(image, kernels, reduce);
}

int Transformer::getPadding(int filter, int input, int stride) {
    // O = I = (I - F + 2P) / S + 1
    // P = ((I - 1) * S - I + F) / 2
    return ((input - 1) * stride - input + filter) / 2;
}

cv::Mat Transformer::broadcast(const cv::Mat &src, int n_channel) {
    cv::Mat dst;
    cv::Mat m = cv::Mat::ones(n_channel, 1, src.type());
    cv::transform(src, dst, m);
    return dst;
}

void Transformer::setParallelMode(bool mode) {
    this->parallel = mode;
}

void Transformer::permute(cv::Mat &src, int n_channel, int type) {
    if (src.channels() != n_channel) {
        src = broadcast(src.clone(), n_channel);
        CV_Assert(src.isContinuous());
    }

    if (src.type() != type) {
        cv::Mat output;
        src.convertTo(output, type);
        src = output;
    }
}
