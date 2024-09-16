//
//  FeatureExtractor.h
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

#import <Foundation/Foundation.h>
#import <Libtorch-Lite/Libtorch-Lite.h>
#import "opencv2/opencv.hpp"

@interface Net : torch::nn::Module {
    // Define your network layers and parameters here
}

- (torch::Tensor)forward:(torch::Tensor)input;

@end

@interface Extractor : NSObject
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithModelPath:(NSString *)modelPath useCuda:(BOOL)useCuda;
- (NSArray *)extractFeatures:(NSArray *)imageCrops;

@end

