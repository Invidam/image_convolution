#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <fstream>

class Calculator {
public:
    static void calculate(cv::Mat& img) {
        // Assuming img is an 8-bit unsigned image
        img.forEach<cv::Vec3b>([](cv::Vec3b &pixel, const int* position) -> void {
            for (int i = 0; i < 3; ++i) {
                pixel[i] =  255 - pixel[i];
            }
        });
    }
};

int main() {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        perror("getcwd() error");
        return 1;
    }

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "Start image convolution" << std::endl;
    std::string inputPath = "images/input.jpg"; // Adjust path as necessary
// Check if the file can be opened with standard C++ I/O
    std::ifstream file(inputPath);
    if (!file) {
        std::cerr << "Failed to open file with ifstream: " << inputPath << std::endl;
    } else {
        std::cout << "File exists and can be opened: " << inputPath << std::endl;
    }

// Attempt to open with OpenCV
    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image with OpenCV: " << inputPath << std::endl;
    } else {
        std::cout << "Image loaded successfully." << std::endl;
    }

    Calculator::calculate(image);

    if (!cv::imwrite("output.jpg", image)) {
        std::cerr << "Failed to save the image" << std::endl;
        return -1;
    }

    std::cout << "Image processed and saved as output.png" << std::endl;
    return 0;
}
