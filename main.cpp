#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "Calculator.h"
#include "Timer.h"

namespace fs = std::filesystem;

std::string selectImageFromFolder() {
    std::string imagesPath = "images/";
    std::vector<std::string> imageFiles;

    // List all files in the directory and filter for images
    std::cout << "Listing all images in directory: " << imagesPath << std::endl;
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
        std::cerr << "No image files found in the directory." << std::endl;
        return "";
    }

    // Allow user to select an image
    std::cout << "Enter the number of the image you wish to process: ";
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
    std::cout << "===========\n[1] Load Image.\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    stepTimer.reset();
    cv::Mat result = Calculator::calculate(image);
    std::cout << "===========\n[2] Convolution Image.\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    stepTimer.reset();
    if (!cv::imwrite("output.jpg", result)) {
        std::cerr << "Failed to save the image." << std::endl;
        return -1;
    }
    std::cout << "===========\n[3] Write Image.\nElapsed time: " << stepTimer.elapsed() << " ms" << std::endl;

    std::cout << "===========\n[F] Full process.\nElapsed time: " << fullTimer.elapsed() << " ms" << std::endl;
    return 0;
}

int main() {
    std::string selectedImage = selectImageFromFolder();
    if (selectedImage.empty()) {
        return -1;
    }

    return measureBySteps(selectedImage);
}
