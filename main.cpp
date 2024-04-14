#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include <functional>
#include "Transformer.h"
#include "Timer.h"
#include "Filter.h"

namespace fs = std::filesystem;
namespace cvl = cv::utils::logging;

std::string getExtension(const std::string &filePath) {
    return filePath.substr(filePath.find_last_of('.') + 1);
}

std::string selectImageFromFolder() {
    std::string imagesPath = "images";
    std::vector<std::string> imageFiles;
    std::string extension;
    fs::create_directory(imagesPath);

    // List all files in the directory and filter for images
    std::cout << "Listing all images from: " << imagesPath << std::endl;
    for (const auto &entry: fs::directory_iterator(imagesPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            // Simple check for image file extensions
            if (filePath.size() >= 4 && (filePath.substr(filePath.size() - 4) == ".jpg" ||
                                         filePath.substr(filePath.size() - 4) == ".png" ||
                                         filePath.substr(filePath.size() - 5) == ".jpeg")) {
                imageFiles.push_back(filePath);
                std::cout << imageFiles.size() << ": " << filePath << std::endl;
            }
        }
    }

    if (imageFiles.empty()) {
        std::cerr << "No image found." << std::endl;
        return "";
    }

    // Allow user to select an image
    std::cout << "Enter the number of image to process: ";
    size_t selection;
    std::cin >> selection;
    if (selection < 1 || selection > imageFiles.size()) {
        std::cerr << "Invalid selection." << std::endl;
        return "";
    }

    return imageFiles[selection - 1];
}

int size;
float sigma;
int type;

std::function<cv::Mat()> getFilterCallback(Filter &filter) {
    std::function<cv::Mat()> callback;

    while (true) {
        std::cout << "1. Gaussian blur\n";
        std::cout << "2. Sobel (edge detection)\n";
        std::cout << "...select a filter: ";
        std::cin >> type;

        if (type == 1) {
            std::cout << "Enter filter size: ";
            std::cin >> size;
            std::cout << "Enter sigma: ";
            std::cin >> sigma;
            callback = [=, &filter]() {
                return filter.gaussianBlur(size, sigma);
            };
            break;
        } else if (type == 2) {
            std::cout << "Enter sigma: ";
            std::cin >> sigma;
            callback = [=, &filter]() {
                return filter.sobel(sigma);
            };
            break;
        } else {
            std::cout << "Unknown type!\n";
        }
    }

    return callback;
}

std::function<cv::Mat()> getFilterCallbackV2(Filter &filter) {
    std::function<cv::Mat()> callback;

    while (true) {
        if (type == 1) {
            std::cout << "1. Gaussian blur\n";
            std::cout << "Size: " << size << "\n"
                      << "Sigma: " << sigma << "\n";
            callback = [=, &filter]() {
                return filter.gaussianBlur(size, sigma);
            };
            break;
        } else if (type == 2) {
            std::cout << "2. Sobel (edge detection)\n";
            std::cout << "Sigma: " << sigma << "\n";
            callback = [=, &filter]() {
                return filter.sobel(sigma);
            };
            break;
        } else {
            std::cout << "Unknown type!\n";
        }
    }

    return callback;
}

int measureBySteps(const std::string &selectedImage) {
    Timer fullTimer;
    Timer stepTimer;
    fullTimer.reset();
    stepTimer.reset();

    std::cout << "===========\n[1] Importing image...\n";
    cv::Mat image = cv::imread(selectedImage, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image: " << selectedImage << std::endl;
        return -1;
    }
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n";
    Filter filter(image, false, false);
    auto compute = getFilterCallback(filter);

    std::cout << "===========\n[2-1] Transforming in serial..." << std::endl;
    stepTimer.reset();
    compute();
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[2-2] Transforming in parallel..." << std::endl;
    stepTimer.reset();
    filter.setParallelMode(true);
    auto result = compute();
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[2-3] Transforming by opencv..." << std::endl;
    stepTimer.reset();

    int depth = CV_16S;  // Depth of the output image

    auto result_opencv = type == 1 ? filter.opencvGaussianBlur(sigma, size) : filter.opencvSobel(CV_8U, 1, 0, 3);
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[3] Writing image...\n";
    stepTimer.reset();
    fs::create_directory("output");
    if (!cv::imwrite("output/" + fs::path(selectedImage).filename().string(), result)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    if (!cv::imwrite("output/opencv-" + fs::path(selectedImage).filename().string(), result_opencv)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    std::cout << "Target: " << "output/" + fs::path(selectedImage).filename().string() << std::endl;
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\nTotal process time: " << fullTimer.elapsed() << " ms" << std::endl;
    return 0;
}

int measureByStepsV2(const std::string &selectedImage) {
    Timer fullTimer;
    Timer stepTimer;
    fullTimer.reset();
    stepTimer.reset();

    std::cout << "===========\n[1] Importing image...\n";
    cv::Mat image = cv::imread(selectedImage, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image: " << selectedImage << std::endl;
        return -1;
    }
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n";
    Filter filter(image, false, false);
    auto compute = getFilterCallbackV2(filter);

    std::cout << "===========\n[2-1] Transforming in serial..." << std::endl;
    stepTimer.reset();
    compute();
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[2-2] Transforming in parallel..." << std::endl;
    stepTimer.reset();
    filter.setParallelMode(true);
    auto result = compute();
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[2-3] Transforming by opencv..." << std::endl;
    stepTimer.reset();

    int depth = CV_16S;  // Depth of the output image

    auto result_opencv = type == 1 ? filter.opencvGaussianBlur(sigma, size) : filter.opencvSobel(CV_8U, 1, 0, 3);
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[3] Writing image...\n";
    stepTimer.reset();
    fs::create_directory("output");
    if (!cv::imwrite("output/" + fs::path(selectedImage).filename().string(), result)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    if (!cv::imwrite("output/opencv-" + fs::path(selectedImage).filename().string(), result_opencv)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    std::cout << "Target: " << "output/" + fs::path(selectedImage).filename().string() << std::endl;
    std::cout << "Elapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\nTotal process time: " << fullTimer.elapsed() << " ms" << std::endl;
    return 0;
}

int main() {
    std::cout << "# of max threads: " << omp_get_max_threads() << std::endl;
    cvl::setLogLevel(cvl::LogLevel::LOG_LEVEL_WARNING);
    std::string selectedImage = selectImageFromFolder();
    int result;

    if (selectedImage.empty()) {
        result = -1;
    } else {
        int types[2] = {1, 2};
        int sizes[3] = {3, 33, 333};
        float sigma = 1.0f;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                type = types[i];
                size = sizes[j];
                std::cout << "Type: " << type << "\n"
                          << "Size: " << size << "\n"
                          << "Sigma: " << sigma << "\n";
                result = measureByStepsV2(selectedImage);
            }
        }
    }

    std::cout << "\nPress ENTER to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return result;
}
