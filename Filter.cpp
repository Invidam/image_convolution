#include <cmath>
#include "Transformer.h"
#include "Filter.h"
#include "Timer.h"

Filter::Filter(const cv::Mat &image) : Filter(image, false) {
}

Filter::Filter(const cv::Mat &image, bool parallel, bool verbose) {
    this->is_parallel = parallel;
    this->verbose = verbose;
    this->transformer = Transformer(parallel, verbose);
    this->image = cv::Mat(image);
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

// Method to perform Gaussian blur using OpenCV's built-in function
cv::Mat Filter::opencvGaussianBlur(float sigma, int filter_size) const {
    if (filter_size < 3) {
        filter_size = 3;
    }
    if (filter_size % 2 == 0) {
        ++filter_size;  // Ensure the filter size is odd
    }

    cv::Mat result;
    Timer timer;  // Assuming Timer is a utility to measure time
    cv::GaussianBlur(image, result, cv::Size(filter_size, filter_size), sigma, sigma);

    if (verbose) {
        std::cout << "OpenCV GaussianBlur completed in " << timer.elapsed() << " ms" << std::endl;
    }

    return result;
}

// Method to perform Sobel edge detection using OpenCV's built-in function
cv::Mat Filter::opencvSobel(int ddepth, int dx, int dy, int ksize) const {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Mat grad;

    Timer timer;  // Assuming Timer is a utility to measure time

    // Compute gradients on x and y
    cv::Sobel(image, grad_x, ddepth, dx, 0, ksize);
    cv::Sobel(image, grad_y, ddepth, 0, dy, ksize);

    // Calculating the magnitude of gradients
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    if (verbose) {
        std::cout << "OpenCV Sobel completed in " << timer.elapsed() << " ms" << std::endl;
    }

    return grad;
}

void Filter::setParallelMode(bool parallel) {
    this->is_parallel = parallel;
    this->transformer.setParallelMode(parallel);
}
