//
//  KalmanFilter.swift
//  KalmanFilter
//
//  Created by Oleksii on 18/08/16.
//  Copyright © 2016 Oleksii Dykan. All rights reserved.
//

import Foundation

/**
 Conventional Kalman Filter
 */
public struct KalmanFilter<Type: KalmanInput>: KalmanFilterType {
    
    public var x: Type 
    public var P: Type
    public var F: Type
    public var H: Type
    public var R: Type
    public var Q: Type
    public var u: Type
    public var B: Type
    
    /// x̂_k|k-1
    public let stateEstimatePrior: Type
    /// P_k|k-1
    public let errorCovariancePrior: Type
    
    public init(stateEstimatePrior: Type, errorCovariancePrior: Type) {
        self.stateEstimatePrior = stateEstimatePrior
        self.errorCovariancePrior = errorCovariancePrior
        self.x = Matrix(vector: [0, 0, 0, 0, 0, 0, 0]) as! Type
        self.P = Matrix(identityOfSize: 7) as! Type
        self.F = Matrix([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]]) as! Type
        self.H = Matrix([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]]) as! Type
        self.R = Matrix(identityOfSize: 4) as! Type
        self.Q = Matrix(identityOfSize: 7) as! Type
        self.u = Matrix(vector: [0, 0]) as! Type
        self.B = Matrix(rows: 7, columns: 2) as! Type
    }
    
    /**
     Predict step in Kalman filter.
     
     - parameter stateTransitionModel: F_k
     - parameter controlInputModel: B_k
     - parameter controlVector: u_k
     - parameter covarianceOfProcessNoise: Q_k
     
     - returns: Another instance of Kalman filter with predicted x̂_k and P_k
     */
    public func predict(stateTransitionModel: Type, controlInputModel: Type, controlVector: Type, covarianceOfProcessNoise: Type) -> KalmanFilter {
        // x̂_k|k-1 = F_k * x̂_k-1|k-1 + B_k * u_k
        let predictedStateEstimate = stateTransitionModel * stateEstimatePrior + controlInputModel * controlVector
        // P_k|k-1 = F_k * P_k-1|k-1 * F_k^t + Q_k
        let predictedEstimateCovariance = stateTransitionModel * errorCovariancePrior * stateTransitionModel.transposed + covarianceOfProcessNoise
        
        return KalmanFilter(stateEstimatePrior: predictedStateEstimate, errorCovariancePrior: predictedEstimateCovariance)
    }
    
    /**
     Update step in Kalman filter. We update our prediction with the measurements that we make
     
     - parameter measurement: z_k
     - parameter observationModel: H_k
     - parameter covarienceOfObservationNoise: R_k
     
     - returns: Updated with the measurements version of Kalman filter with new x̂_k and P_k
     */
    public func update(measurement: Type, observationModel: Type, covarienceOfObservationNoise: Type) -> KalmanFilter {
        // H_k^t transposed. We cache it improve performance
        let observationModelTransposed = observationModel.transposed
        
        // ỹ_k = z_k - H_k * x̂_k|k-1
        let measurementResidual = measurement - observationModel * stateEstimatePrior
        // S_k = H_k * P_k|k-1 * H_k^t + R_k
        let residualCovariance = observationModel * errorCovariancePrior * observationModelTransposed + covarienceOfObservationNoise
        // K_k = P_k|k-1 * H_k^t * S_k^-1
        let kalmanGain = errorCovariancePrior * observationModelTransposed * residualCovariance.inversed
        
        // x̂_k|k = x̂_k|k-1 + K_k * ỹ_k
        let posterioriStateEstimate = stateEstimatePrior + kalmanGain * measurementResidual
        // P_k|k = (I - K_k * H_k) * P_k|k-1
        let posterioriEstimateCovariance = (kalmanGain * observationModel).additionToUnit * errorCovariancePrior
        
        return KalmanFilter(stateEstimatePrior: posterioriStateEstimate, errorCovariancePrior: posterioriEstimateCovariance)
    }
}
