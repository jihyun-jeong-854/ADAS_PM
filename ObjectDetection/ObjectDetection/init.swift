//
//  init.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/20/23.
//

import Foundation

public func buildTracker(cfg: DeepSortConfig, useCuda: Bool) -> DeepSort {
    return DeepSort(reidCheckpoint: cfg.deepSort.reidCheckpoint,
                    maxDist: cfg.deepSort.maxDist,
                    minConfidence: cfg.deepSort.minConfidence,
                    nmsMaxOverlap: cfg.deepSort.nmsMaxOverlap,
                    maxIouDistance: cfg.deepSort.maxIouDistance,
                    maxAge: cfg.deepSort.maxAge,
                    nInit: cfg.deepSort.nInit,
                    nnBudget: cfg.deepSort.nnBudget,
                    useCuda: useCuda)
}

public struct DeepSortConfig {
    public struct DeepSortParameters {
        public let reidCheckpoint: String
        public let maxDist: Float
        public let minConfidence: Float
        public let nmsMaxOverlap: Float
        public let maxIouDistance: Float
        public let maxAge: Int
        public let nInit: Int
        public let nnBudget: Int
    }

    public let deepSort: DeepSortParameters
    // Add other configurations as needed
    // ...
}
