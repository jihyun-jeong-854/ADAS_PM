//
//  Sort.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/14/23.
//

import Foundation

class Sort : NSObject{
    var maxAge: Int
    var minHits: Int
    var iouThreshold: Double
    var trackers: [KalmanBoxTracker]
    var frameCount: Int
    private var sortProcessor = SortProcessor()
    
    init(maxAge: Int = 1, minHits: Int = 3, iouThreshold: Double = 0.3) {
        self.maxAge = maxAge
        self.minHits = minHits
        self.iouThreshold = iouThreshold
        self.trackers = []
        self.frameCount = 0
    }
    
    func getTrackers() -> [KalmanBoxTracker] {
        return trackers
    }
    
    func update(dets: [[Double]] = [[]]) -> [[Double]] {
        self.frameCount += 1
        
        // Get predicted locations from existing trackers
        var trks = Array(repeating: Array(repeating: 0.0, count: 6), count: self.trackers.count)
        var toDel: [Int] = []
        var ret: [[Double]] = []
        
        for (t, trk) in trks.enumerated() {
            let pos = self.trackers[t].predict()
            var trk = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if pos.contains(where: { $0.isNaN }) {
                toDel.append(t)
            }
        }
        
        var maskedTrks = trks.filter { !$0.contains(where: { $0.isNaN }) }
        for t in toDel.reversed() {
            self.trackers.remove(at: t)
        }
        
        let (matched, unmatchedDets, _) = sortProcessor.associateDetectionsToTrackers(detections: dets, trackers: maskedTrks, iouThreshold: self.iouThreshold)
        
        // Update matched trackers with assigned detections
        for m in matched {
            self.trackers[m[1]].update(bbox: dets[m[0]])
        }
        
        // Create and initialize new trackers for unmatched detections
        for i in unmatchedDets {
            let trk = KalmanBoxTracker(bbox: dets[i])
            self.trackers.append(trk)
        }
        
        var i = self.trackers.count
        for trk in self.trackers.reversed() {
            let d = trk.getState()[0]
            if (trk.timeSinceUpdate < 1) && (trk.hitStreak >= self.minHits || self.frameCount <= self.minHits) {
                ret.append([d] + [Double(trk.id + 1)])
            }
            i -= 1
            // Remove dead trackletf
            if trk.timeSinceUpdate > self.maxAge {
                self.trackers.remove(at: i)
            }
        }
        
        if ret.count > 0 {
            return ret
        }
        return [[]]
    }
}
