#include "Timer.h"

Timer::Timer() : m_startTime(std::chrono::high_resolution_clock::now()) {}

void Timer::reset() {
    m_startTime = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed() const {
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(endTime - m_startTime).count();
}
