//
//  DeepSort.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

import Foundation

public class DeepSort {
    private var minConfidence: Float
    private var nmsMaxOverlap: Float
    private var extractor: Extractor
    private var tracker: Tracker
    private var height: Int
    private var width: Int
    
    public init(reidCheckpoint: String, maxDist: Float, minConfidence: Float, nmsMaxOverlap: Float, maxIouDistance: Float, maxAge: Int, nInit: Int, nnBudget: Int, useCuda: Bool) {

        self.minConfidence = minConfidence
        self.nmsMaxOverlap = nmsMaxOverlap

        self.extractor = Extractor(modelPath: reidCheckpoint, useCuda: useCuda)

        let maxCosineDistance = maxDist
        let nnBudget = 100
        let metric = NearestNeighborDistanceMetric(metric: "cosine", matchingThreshold: Double(maxCosineDistance), budget: nnBudget)

        self.tracker = Tracker(metric: metric, maxIOUDistance: Double(maxIouDistance), maxAge: maxAge, nInit: nInit)

        self.height = 0
        self.width = 0
    }
    
    public func update(bboxXYWH: [[Float]], confidences: [Float], oriImage: UIImage) -> [[Int]] {
            guard let oriImg = oriImage.cgImage else {
                return []
            }

            self.height = oriImg.height
            self.width = oriImg.width

            let features = self.getFeatures(bboxXYWH: bboxXYWH, oriImage: oriImg)

            let bboxTLWH = self.xywhToTlwh(bboxXYWH: bboxXYWH)
            var bboxTLWHDouble = float2DToDouble2D(float2D: bboxTLWH)
        
            var detections: [Detection] = []
            for i in 0..<confidences.count where confidences[i] > self.minConfidence {
                let detection = Detection(tlwh: bboxTLWHDouble[i], confidence: Double(confidences[i]), feature: features[i] as! [Float])
                detections.append(detection)
            }

            let boxes = detections.map { $0.tlwh }
            let scores = detections.map { $0.confidence }
            
            let boxesFloat = double2DToFloat2D(double2D: boxes)
            let scoresFloat = doubleToFloat(double: scores)
            let indices = nonMaxSuppression(boxes: boxesFloat, maxBboxOverlap: self.nmsMaxOverlap, scores: scoresFloat)
            let selectedDetections = indices.map { detections[$0] }

            self.tracker.predict()
            self.tracker.update(detections: selectedDetections)

            var outputs: [[Int]] = []
            for track in self.tracker.tracks {
                guard track.isConfirmed(), track.timeSinceUpdate <= 1 else {
                    continue
                }

                let box = track.toTLWH()
                let boxFloat = doubleToFloat(double: box)
                let xyxy = self.tlwhToXYXY(tlwh: boxFloat)
                let trackID = track.trackID
                let output = [Int(xyxy[0]), Int(xyxy[1]), Int(xyxy[2]), Int(xyxy[3]), trackID]
                outputs.append(output)
            }

            return outputs
        }
    
    public func doubleToFloat(double: [Double]) -> [Float] {
        var float: [Float] = []
        for i in 0..<double.count {
            float[i] = Float(double[i])
        }
        return float
    }
    
    public func floatToDouble(float: [Float]) -> [Double] {
        var double: [Double] = []
        for i in 0..<float.count {
            double[i] = Double(float[i])
        }
        return double
    }
    
    public func double2DToFloat2D(double2D: [[Double]]) -> [[Float]]{
        var float2D: [[Float]] = [[]]
        for i in 0..<double2D.count {
            for j in 0..<double2D[i].count {
                float2D[i][j] = Float(double2D[i][j])
            }
        }
        return float2D
    }
    
    public func float2DToDouble2D(float2D: [[Float]]) -> [[Double]]{
        var double2D: [[Double]] = [[]]
        for i in 0..<float2D.count {
            for j in 0..<float2D[i].count {
                double2D[i][j] = Double(float2D[i][j])
            }
        }
        return double2D
    }
    
    private func xywhToTlwh(bboxXYWH: [[Float]]) -> [[Float]] {
            return bboxXYWH.map { box in
                var bboxTLWH = box
                bboxTLWH[0] = box[0] - box[2] / 2.0
                bboxTLWH[1] = box[1] - box[3] / 2.0
                return bboxTLWH
            }
        }
    
    private func tlwhToXYXY(tlwh: [Float]) -> [Float] {
            let x1 = max(Int(tlwh[0]), 0)
            let y1 = max(Int(tlwh[1]), 0)
            let x2 = min(Int(tlwh[0] + tlwh[2]), self.width - 1)
            let y2 = min(Int(tlwh[1] + tlwh[3]), self.height - 1)
            return [Float(x1), Float(y1), Float(x2), Float(y2)]
        }
    
    private func getFeatures(bboxXYWH: [[Float]], oriImage: CGImage) -> NSMutableArray {
            var imCrops: [UIImage] = []
            for box in bboxXYWH {
                let xyxy = self.xywhToXYXY(bboxXYWH: box)
                let xyxyToRect = CGRect(x: Double(xyxy[0]), y: Double(xyxy[1]), width: Double(xyxy[2] - xyxy[0]), height: Double(xyxy[3] - xyxy[1]))
                if let croppedImage = cropImage(image: oriImage, rect: xyxyToRect) {
                    imCrops.append(croppedImage)
                }
            }

            if !imCrops.isEmpty {
                let features = self.extractor.extractFeatures(imCrops)
                return features as! NSMutableArray
            } else {
                return []
            }
        }
    
    private func xywhToXYXY(bboxXYWH: [Float]) -> [Float] {
            let x = bboxXYWH[0]
            let y = bboxXYWH[1]
            let w = bboxXYWH[2]
            let h = bboxXYWH[3]
            let x1 = max(Int(x - w / 2), 0)
            let y1 = max(Int(y - h / 2), 0)
            let x2 = min(Int(x + w / 2), self.width - 1)
            let y2 = min(Int(y + h / 2), self.height - 1)
            return [Float(x1), Float(y1), Float(x2), Float(y2)]
        }
    
    private func cropImage(image: CGImage, rect: CGRect) -> UIImage? {
            if let croppedImage = image.cropping(to: rect) {
                return UIImage(cgImage: croppedImage)
            } else {
                return nil
            }
        }
        
    
}

