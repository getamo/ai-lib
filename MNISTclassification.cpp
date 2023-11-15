#include <iostream>
#include <vector>
#include "filereader.cpp"
#include "ainet.h"

int main() {
    //prepare training data
    const std::string MNIST_IMAGES_FILE = "C:\\Users\\User\\projects\\aiproject\\train-images.idx3-ubyte";
    const std::string MNIST_LABELS_FILE = "C:\\Users\\User\\projects\\aiproject\\train-labels.idx1-ubyte";
    std::vector<std::vector<float>> mnistData = readMNISTImages(MNIST_IMAGES_FILE);
    std::vector<std::vector<float>> mnistLabels = readMNISTLabels(MNIST_LABELS_FILE);
    
    //buildnetwork
    Network network;
    network.create_node(0, 0, 28*28);
    network.create_node(1, 0, 16);
    network.create_node(2, 0, 10);
    network.setinput_nodebylayer(0);
    network.setoutput_nodebylayer(2);
    for (auto i = 0; i < 2; i++){
        network.connectall(i, i+1);
    }

    //training
    network.organize_edge();
    int target_epoch = 3;
    std::cout << mnistData.size() << "\n";
    
    for (auto epoch = 0; epoch < target_epoch; epoch++){
        for (unsigned int imagenum = 0; imagenum < 300; imagenum++){
            network.setinputnode_output(mnistData[imagenum]);
            network.feedforward();
            for (auto & confidence: network.getoutput()){
            std::cout << confidence << " ";
            }
            std::cout<<"\n";
            network.stochastic_backpropagate(mnistLabels[imagenum], 1);
            network.clearallinput();
        }
    }
    
    //predicting
    std::cout << "real value: 1" << "\n";
    if (!mnistData.empty()){
        int c = 0;
        for (float pixelValue : mnistData[1]){
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
    network.setinputnode_output(mnistData[1]);
    network.feedforward();
    network.clearallinput();
    for (auto & confidence: network.getoutput()){
        std::cout << confidence << " ";
    }
    std::cout<< "\n\n";
    

    std::cout << "real value: 2" << "\n";
    if (!mnistData.empty()){
        int c = 0;
        for (float pixelValue : mnistData[2]){
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
    network.setinputnode_output(mnistData[2]);
    network.feedforward();
    network.clearallinput();
    for (auto & confidence: network.getoutput()){
        std::cout << confidence << " ";
    }
}
