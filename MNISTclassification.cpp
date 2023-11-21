#include <iostream>
#include <vector>
#include "filereader.cpp"
#include "ainet.h"

int main() {
    //prepare training data
    const std::string MNIST_IMAGES_FILE = "your\\path\\to\\file";
    const std::string MNIST_LABELS_FILE = "your\\path\\to\\file";
    const std::string TEST_IMAGES_FILE = "your\\path\\to\\file";
    const std::string TEST_LABELS_FILE = "your\\path\\to\\file";
    std::vector<std::vector<float>> mnistData = readMNISTImages(MNIST_IMAGES_FILE);
    std::vector<std::vector<float>> mnistLabels = readMNISTLabels(MNIST_LABELS_FILE);
    std::vector<std::vector<float>> testData = readMNISTImages(TEST_IMAGES_FILE);
    std::vector<std::vector<float>> testLabels = readMNISTLabels(TEST_LABELS_FILE);
    
    //buildnetwork
    Network network;
    network.create_node(0, 0, 28*28);
    network.create_node(1, 0, 16);
    network.create_node(2, 0, 16);
    network.create_node(3, 0, 10);
    network.setinput_nodebylayer(0);
    network.setoutput_nodebylayer(3);
    for (auto i = 0; i < 3; i++){
        network.connectall(i, i+1);
    }

    //training
    network.organize_edge();
    int target_epoch = 5;
    std::cout << mnistData.size() << "\n";
    
    for (auto epoch = 0; epoch < target_epoch; epoch++){
        for (unsigned int imagenum = 0; imagenum < 500; imagenum++){
            network.setinputnode_output(mnistData[imagenum]);
            network.feedforward();
            for (auto & confidence: network.getoutput()){
            std::cout << confidence << " ";
            }
            std::cout<<"\n";
            network.stochastic_backpropagate(mnistLabels[imagenum], 0.3);
            network.clearallinput();
        }
    }
    
    //predicting
    std::cout << "real value: 1" << "\n";
    if (!mnistData.empty()){
        int c = 0;
        for (float pixelValue : testData[0]){
            int outpix = (std::round(pixelValue * 10.0f) / 10.0f)*10;
            if (outpix ==10){
                outpix=9;
            }
            std::cout<<outpix;
            if ( ++c % 28 == 0){
                std::cout << "\n";
            }
        }
    }
    std::cout << "predicted value:" << "\n";
    network.setinputnode_output(testData[0]);
    network.feedforward();
    network.clearallinput();
    for (auto & confidence: network.getoutput()){
        std::cout << confidence << " ";
    }
    std::cout<< "\n\n";
    

    std::cout << "real value: 2" << "\n";
    if (!mnistData.empty()){
        int c = 0;
        for (float pixelValue : testData[1]){
            int outpix = (std::round(pixelValue * 10.0f) / 10.0f)*10;
            if (outpix ==10){
                outpix=9;
            }
            std::cout<<outpix;
            if ( ++c % 28 == 0){
                std::cout << "\n";
            }
        }
    }
    std::cout << "predicted value:" << "\n";
    network.setinputnode_output(testData[1]);
    network.feedforward();
    network.clearallinput();
    for (auto & confidence: network.getoutput()){
        std::cout << confidence << " ";
    }
}
