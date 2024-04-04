//
// Created by Hansu Park on 2024/04/04.
//

#include "Calculator.h"

void Calculator::calculate(cv::Mat& img) {
    img.forEach<cv::Vec3b>([](cv::Vec3b &pixel, const int*) -> void {
        for (int i = 0; i < 3; ++i) {
            pixel[i] = 255 - pixel[i];
        }
    });
}
