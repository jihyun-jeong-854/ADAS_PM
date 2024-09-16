//
//  SortProcessor.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/14/23.
//

import Foundation

class SortProcessor : NSObject {
    
    func linearAssignment(costMatrix: [[Double]]) -> [[Int]] {
        do {
            guard let lapModule = NSClassFromString("lap") as? NSObject.Type else {
                throw ImportError.moduleNotFound("lap")
            }
            
            let lapjvSelector = NSSelectorFromString("lapjv:extend_cost:")
            if lapModule.responds(to: lapjvSelector) {
                let lapjvMethod = lapModule.method(for: lapjvSelector)
                let lapjvFunction: @convention(c) ([Any], Bool) -> [Any] 
                lapjvFunction = unsafeBitCast(lapjvMethod, to: type(of: lapjvFunction))
                
                let result = lapjvFunction(costMatrix.compactMap { $0 }, true)
                let x = result[1] as! [Int]
                let y = result[2] as! [Int]
                
                let pairs = zip(x, y).filter { $0.0 >= 0 }.map { [ $0.1, $0.0 ] }
                return pairs
            } else {
                throw ImportError.functionNotFound("lapjv")
            }
        } catch ImportError.moduleNotFound(let moduleName) {
            print("ImportError: Module '\(moduleName)' not found.")
        } catch ImportError.functionNotFound(let functionName) {
            print("ImportError: Function '\(functionName)' not found.")
        } catch {
            print("An error occurred: \(error)")
        }
        
        // Fallback to scipy.optimize.linear_sum_assignment if lapjv is not available
        let linearSumAssignmentResult = linearSumAssignment(costMatrix: costMatrix)
        return linearSumAssignmentResult
    }

    func linearSumAssignment(costMatrix: [[Double]]) -> [[Int]] {
        var hungAlg = HungarianAlgorithm(inputMatrix: costMatrix)
        let result = hungAlg.solve()
        return result as! [[Int]]
    }

    enum ImportError: Error {
        case moduleNotFound(String)
        case functionNotFound(String)
    }
    

    func iouBatch(bbTest: [[Double]], bbGT: [[Double]]) -> [[Double]] {
        var bbGTExpanded = bbGT
        bbGTExpanded.insert([], at: 0)
        var bbTestExpanded = bbTest
        bbTestExpanded.insert([], at: 0)

        var result = [[Double]]()
     
        for i in 0..<bbTest.count {
            var subArray = [Double]()
            for j in 0..<bbGT.count {
                let xx1 = max(bbTest[i][0], bbGT[j][0])
                let yy1 = max(bbTest[i][1], bbGT[j][1])
                let xx2 = min(bbTest[i][2], bbGT[j][2])
                let yy2 = min(bbTest[i][3], bbGT[j][3])

                let w = max(0.0, xx2 - xx1)
                let h = max(0.0, yy2 - yy1)
                let wh = w * h

                let areaTest = (bbTest[i][2] - bbTest[i][0]) * (bbTest[i][3] - bbTest[i][1])
                let areaGT = (bbGT[j][2] - bbGT[j][0]) * (bbGT[j][3] - bbGT[j][1])
                let overlap = wh / (areaTest + areaGT - wh)
                
                subArray.append(overlap)
            }
            result.append(subArray)
        }
        return result
    }

    func convertBboxToZ(bbox: [Double]) -> Matrix {
        let w = bbox[2] - bbox[0]
        let h = bbox[3] - bbox[1]
        let x = bbox[0] + w / 2.0
        let y = bbox[1] + h / 2.0
        let s = w * h
        let r = w / h
        
        return Matrix(vector: [x, y, s, r])
    }
    
    func convertXToBbox(x: [Double], score: Double? = nil) -> [Double] {
        let w = sqrt(x[2] * x[3])
        let h = x[2] / w
        
        if let scoreValue = score {
            return [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, scoreValue]
        } else {
            return [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        }
    }
    
    func iou(box1: [Double], box2: [Double]) -> Double {
        let x1 = max(box1[0], box2[0])
        let y1 = max(box1[1], box2[1])
        let x2 = min(box1[2], box2[2])
        let y2 = min(box1[3], box2[3])

        let w = max(0.0, x2 - x1)
        let h = max(0.0, y2 - y1)

        let intersectionArea = w * h

        let box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        let box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        let iouValue = intersectionArea / (box1Area + box2Area - intersectionArea)

        return iouValue
    }

    func associateDetectionsToTrackers(detections: [[Double]], trackers: [[Double]], iouThreshold: Double = 0.3) -> ([[Int]], [Int], [Int]) {
        if trackers.isEmpty {
            return ([], Array(0..<detections.count), [])
        }

        let iouMatrix = iouBatch(bbTest: detections, bbGT: trackers)

        var matchedIndices: [[Int]]

        if min(iouMatrix.count, iouMatrix[0].count) > 0 {
            let a = iouMatrix.map { row in row.map { $0 > iouThreshold ? 1 : 0 } }
            if a.map({ $0.reduce(0, +) }).max() == 1 && a.reduce(0, { max($0, $1.reduce(0, +)) }) == 1 {
                        matchedIndices = a.enumerated().flatMap { (i, row) in row.enumerated().map { (j, value) in value == 1 ? [i, j] : [] } }
                    } else {
                        matchedIndices = linearAssignment(costMatrix: iouMatrix)
            }
        } else {
            matchedIndices = []
        }

        var unmatchedDetections = [Int]()
        for d in 0..<detections.count {
            if !matchedIndices.map({ $0[0] }).contains(d) {
                unmatchedDetections.append(d)
            }
        }

        var unmatchedTrackers = [Int]()
        for t in 0..<trackers.count {
            if !matchedIndices.map({ $0[1] }).contains(t) {
                unmatchedTrackers.append(t)
            }
        }

        var matches = [[Int]]()
        for m in matchedIndices {
            let iouValue = iouMatrix[m[0]][m[1]]
            if iouValue < iouThreshold {
                unmatchedDetections.append(m[0])
                unmatchedTrackers.append(m[1])
            } else {
                matches.append(m)
            }
        }

        return (matches, unmatchedDetections, unmatchedTrackers)
    }
    
}


