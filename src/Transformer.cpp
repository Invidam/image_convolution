#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include "Transformer.h"

        cv::Mat Transformer::convolve(const cv::Mat &image,
                                      const cv::Mat &kernel) const { // NOLINT(*-convert-member-functions-to-static)
    std::vector kernels(1, kernel);
    std::function<int(int[])> reduce;
    return convolve(image, kernels, reduce);
}


// CUDA kernel for convolution
__global__ void convolveKernel(const uchar* i_ptr, uchar* o_ptr, const int* f_ptr, int Fh, int Fw, int Ow, int CH, int Iw, int v_step, int h_step, int K, const int* divisor, const int* sum, bool verbose) {
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

cv::Mat Transformer::convolve(const cv::Mat &image, const std::vector<cv::Mat> &kernels, const std::function<int(int[])> &reduce) const {
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
    if(!is_cuda) {
#pragma omp parallel for schedule(static) if(parallel) num_threads(128)
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
                    term[k] = std::abs(sum[k * CH + ch] / divisor[k]);
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

    // Allocate memory on GPU
    uchar *d_i_ptr, *d_o_ptr;
    int *d_f_ptr, *d_divisor, *d_sum;
    size_t img_size = Ih * Iw * CH * sizeof(uchar);
    size_t kernel_size = Fh * Fw * CH * sizeof(int);
    size_t divisor_size = K * sizeof(int);
    cudaMalloc((void**)&d_i_ptr, img_size);
    cudaMalloc((void**)&d_o_ptr, Oh * Ow * CH * sizeof(uchar));
    cudaMalloc((void**)&d_f_ptr, K * kernel_size);
    cudaMalloc((void**)&d_divisor, divisor_size);
    cudaMalloc((void**)&d_sum, K * CH * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_i_ptr, i_ptr, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_ptr, f_ptr[0], K * kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_divisor, divisor.data(), divisor_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int block_size = 256;
    int num_blocks = (v_step * h_step + block_size - 1) / block_size;

    // Launch kernel
    convolveKernel<<<num_blocks, block_size, K * CH * sizeof(int)>>>(d_i_ptr, d_o_ptr, d_f_ptr, Fh, Fw, Ow, CH, Iw, v_step, h_step, K, d_divisor, d_sum, verbose);

    // Copy output data back to host
    cudaMemcpy(o_ptr, d_o_ptr, Oh * Ow * CH * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_i_ptr);
    cudaFree(d_o_ptr);
    cudaFree(d_f_ptr);
    cudaFree(d_divisor);
    cudaFree(d_sum);

    return output;
}

Transformer::Transformer(bool parallel, bool verbose) {
    this->parallel = parallel;
    this->verbose = verbose;
}

void Transformer::setParallelMode(bool mode) {
    this->parallel = mode;
}

void Transformer::setCudaMode(bool isCuda) {
    this->is_cuda = isCuda;
}

int Transformer::getPadding(int filter, int input, int stride) {
    // O = I = (I - F + 2P) / S + 1
    // P = ((I - 1) * S - I + F) / 2
    return ((input - 1) * stride - input + filter) / 2;
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

cv::Mat Transformer::broadcast(const cv::Mat &src, int n_channel) {
    cv::Mat dst;
    cv::Mat m = cv::Mat::ones(n_channel, 1, src.type());
    cv::transform(src, dst, m);
    return dst;
}
