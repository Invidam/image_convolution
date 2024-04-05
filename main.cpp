#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "Calculator.h"
#include "Timer.h"

namespace fs = std::filesystem;
namespace cvl = cv::utils::logging;

std::string getExtension(const std::string &filePath) {
    return filePath.substr(filePath.find_last_of('.') + 1);
}

std::string selectImageFromFolder() {
    std::string imagesPath = "images/";
    std::vector<std::string> imageFiles;
    std::string extension;

    // List all files in the directory and filter for images
    std::cout << "Listing all images from directory: " << imagesPath << std::endl;
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

int measureBySteps(std::string selectedImage) {
    Timer fullTimer;
    Timer stepTimer;
    fullTimer.reset();
    stepTimer.reset();

    // Load and process the imagec
    cv::Mat image = cv::imread(selectedImage, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image: " << selectedImage << std::endl;
        return -1;
    }
    std::cout << "===========\n[1] Importing image...\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    stepTimer.reset();
    cv::Mat result = Calculator::calculate(image);
    std::cout << "===========\n[2] Applying convolution...\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    stepTimer.reset();
    fs::create_directory("output");
    std::string outputFilename = "output/" + fs::path(selectedImage).filename().string();
    std::cout << "Writing output to: " << outputFilename << std::endl;
    if (!cv::imwrite(outputFilename, result)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    std::cout << "===========\n[3] Writing image...\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

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
        result = measureBySteps(selectedImage);
    }

    std::cout << "\nPress ENTER to exit...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get();
    return result;
}
