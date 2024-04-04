//
// Created by Hansu Park on 2024/04/04.
//

#include "Calculator.h"

cv::Mat Calculator::calculate(const cv::Mat &img) {
    cv::Mat result = img.clone(); // Create a copy of the input image

    // Apply the transformation to each pixel of the copied image
    result.forEach<cv::Vec3b>([](cv::Vec3b &pixel, const int *) -> void {
        for (int i = 0; i < 3; ++i) {
            pixel[i] = 255 - pixel[i]; // Invert the pixel value
        }
    });

    return result; // Return the modified image
}
