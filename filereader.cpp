#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <tgmath.h>
// MNIST dataset file paths

int swapEndian(int value) {
    return ((value >> 24) & 0x000000FF) | ((value >> 8) & 0x0000FF00) |
           ((value << 8) & 0x00FF0000) | ((value << 24) & 0xFF000000);
}

// Function to read MNIST images and return a vector of float values
std::vector<std::vector<float>> readMNISTImages(const std::string& imagesFile) {
    std::ifstream file(imagesFile, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open MNIST images file." << std::endl;
        return {};
    }

    int magicNumber, numImages, numRows, numCols;

    // Read the magic number in big-endian format and swap byte order
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = swapEndian(magicNumber);

    file.read((char*)&numImages, sizeof(numImages));
    numImages = swapEndian(numImages);

    file.read((char*)&numRows, sizeof(numRows));
    numRows = swapEndian(numRows);

    file.read((char*)&numCols, sizeof(numCols));
    numCols = swapEndian(numCols);

    // Check if the format of the dataset matches expectations
    if (magicNumber != 2051) {
        std::cerr << "Error: Invalid MNIST images file format." << std::endl;
        return {};
    }

    std::vector<std::vector<float>> mnistData;
    mnistData.reserve(numImages);

    for (int i = 0; i < numImages; ++i) {
        std::vector<float> pixelData;
        pixelData.reserve(numRows * numCols);

        for (int j = 0; j < numRows * numCols; ++j) {
            uint8_t pixelValue;
            file.read((char*)&pixelValue, sizeof(pixelValue));
            float normalizedValue = static_cast<float>(pixelValue) / 255.0f; // Normalize to [0, 1]
            // Round to three decimal places
            normalizedValue = std::round(normalizedValue * 1000.0f) / 1000.0f;
            pixelData.push_back(normalizedValue);
        }

        mnistData.push_back(pixelData);
    }

    file.close();
    return mnistData;
}

std::vector<std::vector<float>> readMNISTLabels(const std::string& labelsFile) {
    std::ifstream file(labelsFile, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open MNIST labels file." << std::endl;
        return {};
    }

    int magicNumber, numLabels;

    // Read the magic number in big-endian format and swap byte order
    file.read((char*)&magicNumber, sizeof(magicNumber));
    magicNumber = swapEndian(magicNumber);

    file.read((char*)&numLabels, sizeof(numLabels));
    numLabels = swapEndian(numLabels);

    // Check if the format of the dataset matches expectations
    if (magicNumber != 2049) {
        std::cerr << "Error: Invalid MNIST labels file format." << std::endl;
        return {};
    }

    std::vector<std::vector<float>> mnistLabels;
    mnistLabels.reserve(numLabels);

    for (int i = 0; i < numLabels; ++i) {
        uint8_t label;
        file.read((char*)&label, sizeof(label));
        std::vector<float> outvector;
        for (auto i=0; i < 10; i++){
            outvector.push_back(0);
        }
        outvector[static_cast<int>(label)] = static_cast<float>(1);
        mnistLabels.push_back(outvector);
    }
    file.close();
    return mnistLabels;
}
