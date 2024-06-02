#include <cmath>
#include "Filter.h"

// Device function for saturate cast
__device__ int saturate_cast_device(double value) {
    return (value > INT_MAX) ? INT_MAX : ((value < INT_MIN) ? INT_MIN : static_cast<int>(value));
}

// CUDA kernel for creating the Gaussian filter
__global__ void gaussianKernel(int* filter, int n_row, int n_col, int origin_row, int origin_col, double ssq, double pi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_row * n_col) {
        int row = i / n_col;
        int col = i % n_col;
        int y = row - origin_row;
        int x = col - origin_col;
        double index = -(x * x + y * y) / ssq;
        filter[i] = saturate_cast_device(UINT8_MAX * exp(index) / (pi * ssq));
    }
}

// CUDA kernel for Sobel convolution
__global__ void sobelKernel(const uchar* i_ptr, uchar* o_ptr, const int* f_ptr, int Fh, int Fw, int Ow, int CH, int Iw, int v_step, int h_step, int K, const int* divisor, const int* sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < v_step * h_step) {
        const int row = i / h_step;
        const int col = i % h_step;
        int start = (row * Iw + col) * CH;
        extern __shared__ int shared_sum[];

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
                    atomicAdd(&shared_sum[k * CH + ch], color * f_ptr[k * Fh * Fw * CH + f_idx + ch]);
                }
            }
        }

        // Point (row+1, col+1) is where the center of kernel overlaps.
        // Let's update this particular pixel for output.
        start = (row * Ow + col) * CH;
        int o_idx = start + (Ow + 1) * CH;

        for (int ch = 0; ch < CH; ++ch) {
            int color = 0;
            for (int k = 0; k < K; ++k) {
                color += abs(shared_sum[k * CH + ch] / divisor[k]);
            }
            o_ptr[o_idx + ch] = color;
        }
    }
}

Filter::Filter(const cv::Mat &image) : Filter(image, false) {}

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
    if(!is_cuda) {
        // Each thread produces roughly 36963 digits per millisecond
        [[maybe_unused]] const int n_thread = std::max(1, is_parallel * n_row * n_col / 36963);
        Timer timer;

#pragma omp parallel for num_threads(128)
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

    // Allocate memory on GPU
    int* d_filter;
    cudaMalloc((void**)&d_filter, n_row * n_col * sizeof(int));

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = (n_row * n_col + block_size - 1) / block_size;

    // Launch kernel to create the Gaussian filter
    gaussianKernel<<<num_blocks, block_size>>>(d_filter, n_row, n_col, origin_row, origin_col, ssq, pi);

    // Copy filter data back to host
    cudaMemcpy(ptr, d_filter, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_filter);

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

    auto reduce = [=] __device__ (const int arr[]) {
        double color = std::sqrt(arr[0] * arr[0] + arr[1] * arr[1]);
        return saturate_cast_device(color);
    };

    auto blur_image = gaussianBlur(3, sigma);
    if(!is_cuda) {
        return transformer.convolve(blur_image, kernel, reduce);
    }
    // Allocate memory on GPU
    uchar *d_i_ptr, *d_o_ptr;
    int *d_f_ptr, *d_divisor, *d_sum;
    size_t img_size = blur_image.rows * blur_image.cols * blur_image.channels() * sizeof(uchar);
    size_t kernel_size = 3 * 3 * blur_image.channels() * sizeof(int);
    size_t divisor_size = 2 * sizeof(int);
    cudaMalloc((void**)&d_i_ptr, img_size);
    cudaMalloc((void**)&d_o_ptr, img_size);
    cudaMalloc((void**)&d_f_ptr, 2 * kernel_size);
    cudaMalloc((void**)&d_divisor, divisor_size);
    cudaMalloc((void**)&d_sum, 2 * blur_image.channels() * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_i_ptr, blur_image.ptr<uchar>(), img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_ptr, kernel[0].ptr<int>(), 2 * kernel_size, cudaMemcpyHostToDevice);
    int divisor[2] = {1, 1}; // Example divisor values
    cudaMemcpy(d_divisor, divisor, divisor_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = ((blur_image.rows - 2) * (blur_image.cols - 2) + block_size - 1) / block_size;

    // Launch kernel to apply the Sobel filter
    sobelKernel<<<num_blocks, block_size, 2 * blur_image.channels() * sizeof(int)>>>(d_i_ptr, d_o_ptr, d_f_ptr, 3, 3, blur_image.cols - 2, blur_image.channels(), blur_image.cols, blur_image.rows - 2, blur_image.cols - 2, 2, d_divisor, d_sum);

    // Copy output data back to host
    cudaMemcpy(blur_image.ptr<uchar>(), d_o_ptr, img_size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_i_ptr);
    cudaFree(d_o_ptr);
    cudaFree(d_f_ptr);
    cudaFree(d_divisor);
    cudaFree(d_sum);

    return blur_image;
}


void Filter::setParallelMode(bool parallel) {
    this->is_parallel = parallel;
    this->transformer.setParallelMode(parallel);
}

void Filter::setCudaMode(bool isCuda) {
    this->is_cuda = isCuda;
    this->transformer.setParallelMode(isCuda);
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

    // Compute gradients on x and
    // y
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
