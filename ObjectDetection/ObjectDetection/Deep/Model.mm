//
//  Model.mm
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

#import <Foundation/Foundation.h>
#import <Libtorch-Lite/Libtorch-Lite.h>

class BasicBlock : public torch::nn::Module {
public:
    BasicBlock(int cIn, int cOut, bool isDownsample) : isDownsample(isDownsample) {
        if (isDownsample) {
            conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 3).stride(2).padding(1).bias(false)));
        } else {
            conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 3).stride(1).padding(1).bias(false)));
        }
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(cOut));
        relu = register_module("relu", torch::nn::ReLU(true));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(cOut, cOut, 3).stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(cOut));
        if (isDownsample) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 1).stride(2).bias(false)),
                torch::nn::BatchNorm2d(cOut)
            ));
        } else if (cIn != cOut) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(cIn, cOut, 1).stride(1).bias(false)),
                torch::nn::BatchNorm2d(cOut)
            ));
            isDownsample = true;
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor y = conv1->forward(x);
        y = bn1->forward(y);
        y = relu->forward(y);
        y = conv2->forward(y);
        y = bn2->forward(y);
        if (isDownsample) {
            x = downsample->forward(x);
        }
        return torch::relu(x.add(y), true);
    }

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::Sequential downsample{nullptr};
    bool isDownsample;
};

torch::nn::Sequential makeLayers(int cIn, int cOut, int repeatTimes, bool isDownsample = false) {
    torch::nn::Sequential layers;
    for (int i = 0; i < repeatTimes; ++i) {
        if (i == 0) {
            layers->push_back(BasicBlock(cIn, cOut, isDownsample));
        } else {
            layers->push_back(BasicBlock(cOut, cOut));
        }
    }
    return layers;
}

class Net : public torch::nn::Module {
public:
    Net(int numClasses = 751, bool reid = false) : reid(reid) {
        conv = register_module("conv", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(true),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))
        ));
        layer1 = register_module("layer1", makeLayers(64, 64, 2, false));
        layer2 = register_module("layer2", makeLayers(64, 128, 2, true));
        layer3 = register_module("layer3", makeLayers(128, 256, 2, true));
        layer4 = register_module("layer4", makeLayers(256, 512, 2, true));
        avgpool = register_module("avgpool", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({8, 4}).stride(1)));
        classifier = register_module("classifier", torch::nn::Sequential(
            torch::nn::Linear(512, 256),
            torch::nn::BatchNorm1d(256),
            torch::nn::ReLU(true),
            torch::nn::Dropout(),
            torch::nn::Linear(256, numClasses)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        if (reid) {
            x = x.div(x.norm(2, 1, true));
            return x;
        }
        x = classifier->forward(x);
        return x;
    }

private:
    torch::nn::Sequential conv{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier{nullptr};
    bool reid;
};
