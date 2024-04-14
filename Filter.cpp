#include <cmath>
#include "Transformer.h"
#include "Filter.h"
#include "Timer.h"

Filter::Filter(const cv::Mat &image) : Filter(image, false) {
}

Filter::Filter(const cv::Mat &image, bool parallel) {
    this->transformer = Transformer();
    this->image = cv::Mat(image);
    this->is_parallel = parallel;
}

cv::Mat Filter::gaussianBlur(int filter_size, float sigma) const {
    if (filter_size < 3) {
        filter_size = 3;
    }
    if (filter_size % 2 == 0) {
        ++filter_size;  // make sure it's an odd-number
    }

    const static double pi = acos(-1);
    const int n_row = filter_size;
    const int n_col = filter_size;
    const int origin_row = n_row / 2;
    const int origin_col = n_col / 2;
    const double ssq = 2 * sigma * sigma;
    cv::Mat filter(filter_size, filter_size, CV_32S);
    int *ptr = filter.ptr<int>(0);

    // Each thread produces roughly 36963 digits per millisecond
    [[maybe_unused]] const int n_thread = std::max(1, is_parallel * n_row * n_col / 36963);
    Timer timer;

    #pragma omp parallel for num_threads(n_thread)
    for (int i = 0; i < n_row * n_col; ++i) {
        int row = i / n_col;
        int col = i % n_col;
        int y = row - origin_row;
        int x = col - origin_col;
        double index = -(x * x + y * y) / ssq;
        ptr[i] = cv::saturate_cast<int>(UINT8_MAX * exp(index) / (pi * ssq));
    }

    if (verbose) {
        std::cout << "Filter created in " << timer.elapsed() << " ms" << std::endl;
    }

    return transformer.convolve(image, filter);
}

cv::Mat Filter::sobel(float sigma) const {
    int data[2][9] = {
            {
                    1, 0, -1,
                    2, 0, -2,
                    1, 0, -1
            },
            {
                    -1, -2, -1,
                    0, 0, 0,
                    1, 2, 1
            }
    };

    std::vector<cv::Mat> kernel(2);
    kernel[0] = cv::Mat(3, 3, CV_32S, data[0]);
    kernel[1] = cv::Mat(3, 3, CV_32S, data[1]);

    auto reduce = [=](const int arr[]) {
        double color = std::sqrt(arr[0] * arr[0] + arr[1] * arr[1]);
        return cv::saturate_cast<int>(color);
    };

    auto blur_image = gaussianBlur(3, sigma);
    return transformer.convolve(blur_image, kernel, reduce);
}

void Filter::setParallelMode(bool parallel) {
    this->is_parallel = parallel;
}
