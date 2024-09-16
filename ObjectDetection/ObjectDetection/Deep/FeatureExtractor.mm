//
//  FeatureExtractor.mm
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

#import <Foundation/Foundation.h>
#import <Libtorch-Lite/Libtorch-Lite.h>
#import "opencv2/opencv.hpp"
//#import "FeatureExtractor.h"
@interface Net : torch::nn::Module {
    // Define your network layers and parameters here
}

- (torch::Tensor)forward:(torch::Tensor)input {
    // Implement the forward pass of your network here
}

@end

@interface Extractor : NSObject

- (instancetype)initWithModelPath:(NSString *)modelPath useCuda:(BOOL)useCuda;
- (NSArray *)extractFeatures:(NSArray *)imageCrops;

@end

@implementation Extractor {
    Net _net;
    torch::Device _device;
    cv::Size _size;
    torch::Transform _norm;
}

- (instancetype)initWithModelPath:(NSString *)modelPath useCuda:(BOOL)useCuda {
    self = [super init];
    if (self) {
        _net = Net();
        _device = useCuda && torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
        torch::load(_net, modelPath.UTF8String);
        _net->to(_device);
        NSLog(@"Loading weights from %@... Done!", modelPath);
        _size = cv::Size(64, 128);
        _norm = torch::data::transforms::Normalize<>({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    }
    return self;
}

- (NSArray *)extractFeatures:(NSArray *)imageCrops {
    NSMutableArray *features = [NSMutableArray array];
    for (cv::Mat imageCrop in imageCrops) {
        cv::Mat resizedImage;
        cv::resize(imageCrop, resizedImage, _size);
        resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
        
        torch::Tensor input = torch::from_blob(resizedImage.data, {1, resizedImage.rows, resizedImage.cols, 3}).permute({0, 3, 1, 2}).to(_device);
        input = _norm(input);
        
        torch::NoGradGuard noGrad;
        torch::Tensor output = _net->forward(input);
        output = output.to(torch::kCPU);
        features.push_back(output);
    }
    return [features copy];
}

@end
