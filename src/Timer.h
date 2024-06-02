//
// Created by Hansu Park on 2024/04/05.
//

#ifndef IMAGE_CONVOLUTION_TIMER_H
#define IMAGE_CONVOLUTION_TIMER_H


#include <chrono>

class Timer {
public:
    Timer();

    void reset();

    double elapsed() const;

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};


#endif //IMAGE_CONVOLUTION_TIMER_H
