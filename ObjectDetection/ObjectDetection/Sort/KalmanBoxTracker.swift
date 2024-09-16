//
//  KalmanBoxTracker.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/14/23.
//

import Foundation

class KalmanBoxTracker : NSObject{
    private var sort = SortProcessor()
    static var count = 0
    
    var kf = KalmanFilter(stateEstimatePrior: Matrix(vector: [0, 0, 0, 0, 0, 0, 0]), errorCovariancePrior: Matrix(identityOfSize: 4))
    var timeSinceUpdate: Int
    var id: Int
    var history: [[Double]] = []
    var hits: Int
    var hitStreak: Int
    var age: Int
    var centroidArr: [(Double, Double)]
    var detClass: Double

    init(bbox: [Double]) {
        let dim_x = 7
        let dim_z = 4
        let x = Matrix(vector: [0, 0, 0, 0, 0, 0, 0])
        let P = Matrix(identityOfSize: dim_x)
        self.kf = KalmanFilter(stateEstimatePrior: x, errorCovariancePrior: P)
        self.kf.F = Matrix([[1, 0, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = Matrix([[1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]])
        
//        self.kf.R[2..., 2...] *= 10.0
        self.kf.R = Matrix(identityOfSize: dim_z)
        self.kf.R[2, 2] *= 10.0
        self.kf.R[3, 3] *= 10.0
        
//        self.kf.P[4..., 4...] *= 1000.0
        self.kf.P[4, 4] *= 1000.0
        self.kf.P[5, 5] *= 1000.0
        self.kf.P[6, 6] *= 1000.0
        
//        self.kf.P *= 10.0
        for i in 0..<7 {
            for j in 0..<7 {
                self.kf.P[i,j] *= 10.0
            }
        }
        
//        self.kf.Q[-1, -1] *= 0.5
//        self.kf.Q[4..., 4...] *= 0.5
        self.kf.Q = Matrix(identityOfSize: dim_x)
        self.kf.Q[6, 6] *= 0.5
        self.kf.Q[4, 4] *= 0.5
        self.kf.Q[5, 5] *= 0.5
        self.kf.Q[6, 6] *= 0.5
        
//        self.kf.x[0..<4] = sort.convertBboxToZ(bbox: bbox)
        let bboxToZ = sort.convertBboxToZ(bbox: bbox)
        for i in 0..<dim_z {
            self.kf.x[i, 0] = bboxToZ[i, 0]
        }
        
        
        self.timeSinceUpdate = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hitStreak = 0
        self.age = 0
        self.centroidArr = []
        let cx = (bbox[0] + bbox[2]) / 2.0
        let cy = (bbox[1] + bbox[3]) / 2.0
        self.centroidArr.append((cx, cy))
        self.detClass = bbox[4]
    }

    func update(bbox: [Double]) {
        self.timeSinceUpdate = 0
        self.history = []
        self.hits += 1
        self.hitStreak += 1
        self.kf = self.kf.update(measurement: sort.convertBboxToZ(bbox: bbox), observationModel: self.kf.H, covarienceOfObservationNoise: self.kf.R)
        self.detClass = bbox[5]
        let cx = (bbox[0] + bbox[2]) / 2.0
        let cy = (bbox[1] + bbox[3]) / 2.0
        self.centroidArr.append((cx, cy))
    }

    func predict() -> [Double] {
        if (self.kf.x[6, 0] + self.kf.x[2, 0] <= 0) {
            self.kf.x[6, 0] *= 0.0
        }
        self.kf = self.kf.predict(stateTransitionModel: self.kf.F, controlInputModel: self.kf.B, controlVector: self.kf.u, covarianceOfProcessNoise: self.kf.Q)
        self.age += 1
        if (self.timeSinceUpdate > 0) {
            self.hitStreak = 0
        }
        self.timeSinceUpdate += 1
        
        var xToDouble: [Double] = []
        for i in 0..<7 { xToDouble.append(self.kf.x[i, 0]) }
        self.history.append(sort.convertXToBbox(x: xToDouble))
        var lastHistory = self.history.count-1
        return self.history[lastHistory]
    }

    func getState() -> [Double] {
        let arrDetClass = [self.detClass]

        let arrUDot = [self.kf.x[4, 0]]
        let arrVDot = [self.kf.x[5, 0]]
        let arrSDot = [self.kf.x[6, 0]]
        
        let arrSum = arrDetClass + arrUDot + arrVDot + arrSDot
        
        var xToDouble: [Double] = []
        for i in 0..<7 {xToDouble.append(self.kf.x[i, 0])}
        let result = sort.convertXToBbox(x: xToDouble) + arrSum
        
        return result
    }
}
