//
//  Model.h
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

#import <Foundation/Foundation.h>
#import <Libtorch-Lite/Libtorch-Lite.h>

class BasicBlock : public torch::nn::Module {
public:
    BasicBlock(int cIn, int cOut, bool isDownsample);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Sequential downsample{nullptr};
    bool isDownsample;
};

torch::nn::Sequential makeLayers(int cIn, int cOut, int repeatTimes, bool isDownsample = false);

class Net : public torch::nn::Module {
public:
    Net(int numClasses = 751, bool reid = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential conv{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier{nullptr};
    bool reid;
};
