//
//  ObjectTracker.swift
//  ObjectDetection
//
//  Created by 정지현 on 2023/11/12.
//
import Accelerate
import Foundation
import UIKit
func linear_assignment(cost_matrix: [[Double]]) -> [[Int]] {
    do {
        let lap = try LAPJV(cost_matrix: cost_matrix, extend_cost: true)
        let x = lap.x
        let y = lap.y
        var result: [[Int]] = []
        for i in 0..<x.count {
            if x[i] >= 0 {
                result.append([y[i], i])
            }
        }
        return result
    } catch {
        let linear_sum_assignment = LinearSumAssignment(cost_matrix: cost_matrix)
        let (x, y) = linear_sum_assignment.solve()
        var result: [[Int]] = []
        for i in 0..<x.count {
            result.append([x[i], y[i]])
        }
        return result
    }
}

func iou_batch(bb_test: [[Double]], bb_gt: [[Double]]) -> [Double] {
    var bb_gt_expanded: [[[Double]]] = []
    for _ in 0..<bb_test.count {
        bb_gt_expanded.append(bb_gt)
    }
    var bb_test_expanded: [[[Double]]] = []
    for _ in 0..<bb_gt.count {
        bb_test_expanded.append(bb_test)
    }
    var xx1: [[Double]] = []
    var yy1: [[Double]] = []
    var xx2: [[Double]] = []
    var yy2: [[Double]] = []
    var w: [[Double]] = []
    var h: [[Double]] = []
    var wh: [[Double]] = []
    var o: [Double] = []
    for i in 0..<bb_test.count {
        var xx1_row: [Double] = []
        var yy1_row: [Double] = []
        var xx2_row: [Double] = []
        var yy2_row: [Double] = []
        var w_row: [Double] = []
        var h_row: [Double] = []
        var wh_row: [Double] = []
        for j in 0..<bb_gt.count {
            xx1_row.append(max(bb_test_expanded[i][j][0], bb_gt_expanded[i][j][0]))
            yy1_row.append(max(bb_test_expanded[i][j][1], bb_gt_expanded[i][j][1]))
            xx2_row.append(min(bb_test_expanded[i][j][2], bb_gt_expanded[i][j][2]))
            yy2_row.append(min(bb_test_expanded[i][j][3], bb_gt_expanded[i][j][3]))
            w_row.append(max(0.0, xx2_row[j] - xx1_row[j]))
            h_row.append(max(0.0, yy2_row[j] - yy1_row[j]))
            wh_row.append(w_row[j] * h_row[j])
        }
        xx1.append(xx1_row)
        yy1.append(yy1_row)
        xx2.append(xx2_row)
        yy2.append(yy2_row)
        w.append(w_row)
        h.append(h_row)
        wh.append(wh_row)
        let bb_test_area = (bb_test_expanded[i][0][2] - bb_test_expanded[i][0][0]) * (bb_test_expanded[i][0][3] - bb_test_expanded[i][0][1])
        let bb_gt_area = (bb_gt_expanded[i][0][2] - bb_gt_expanded[i][0][0]) * (bb_gt_expanded[i][0][3] - bb_gt_expanded[i][0][1])
        let o_row = wh_row[0] / (bb_test_area + bb_gt_area - wh_row[0])
        o.append(o_row)
    }
    return o
}

func convert_bbox_to_z(bbox: [Float]) -> [[Float]] {
    let w = bbox[2] - bbox[0]
    let h = bbox[3] - bbox[1]
    let x = bbox[0] + w/2.0
    let y = bbox[1] + h/2.0
    let s = w * h
    
    let r = w / h
    return [[x, y, s, r]]
}

func convert_x_to_bbox(x: [Float], score: Float? = nil) -> [[Float]] {
    let w = sqrt(x[2] * x[3])
    let h = x[2] / w
    if let score = score {
        return [[x[0]-w/2.0, x[1]-h/2.0, x[0]+w/2.0, x[1]+h/2.0, score]]
    } else {
        return [[x[0]-w/2.0, x[1]-h/2.0, x[0]+w/2.0, x[1]+h/2.0]]
    }
}


class Sort_d :NSObject{
    var max_age: Int
    var min_hits: Int
    var frame_count: Int
    var iou_threshold: Float
    static let x = Matrix(vector: [0, 0, 0, 0, 0, 0, 0])
    static let P = Matrix(grid: [1000, 0, 0, 0, 0, 0, 1000], rows: 7, columns: 7)
    static var kf = KalmanFilter(stateEstimatePrior: x, errorCovariancePrior: P)
    var trackers = [kf]
 
    
    init(max_age: Int = 1, min_hits: Int = 3, iou_threshold: Float = 0.3) {
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
    }
    
    func getTrackers()-> [KalmanFilter<stateEstimatePrior: KalmanInput, errorCovariancePrior: KalmanInput>]{
        return self.trackers
    }
    
    func update(dets: [[Float]]) -> [[Float]] {
        self.frame_count += 1
        
        var trks = [[Float]](repeating: [Float](repeating: 0, count: 6), count: self.trackers.count)
        var to_del: [Int] = []
        var ret: [[Float]] = []
        
        for (t, trk) in trks.enumerated() {
            let pos = self.trackers[t].predict()[0]
            trk = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if pos.contains(where: { $0.isNaN }) {
                to_del.append(t)
            }
        }
        
        trks = trks.filter { !to_del.contains(trks.index(of: $0)!) }
        
        let (matched, unmatched_dets, unmatched_trks) = associate_detections_to_trackers(dets: dets, trks: trks, iou_threshold: self.iou_threshold)
        
        for m in matched {
            self.trackers[m[1]].update(dets[m[0]])
        }
        
        for i in unmatched_dets {
            let trk = KalmanBoxTracker(dets[i] + [0])
            self.trackers.append(trk)
        }
        
        var i = self.trackers.count
        for trk in self.trackers.reversed() {
            let d = trk.get_state()[0]
            if (trk.time_since_update < 1) && (trk.hit_streak >= self.min_hits || self.frame_count <= self.min_hits) {
                ret.append(d + [trk.id+1])
            }
            i -= 1
            if trk.time_since_update > self.max_age {
                self.trackers.remove(at: i)
            }
        }
        
        if ret.count > 0 {
            return ret
        }
        
        return [[Float]]()
    }
}

func parse_args() -> Args {
    let parser = ArgumentParser(description: "SORT demo")
    parser.add_argument("--display", dest: "display", help: "Display online tracker output (slow) [False]", action: "store_true")
    parser.add_argument("--seq_path", help: "Path to detections.", type: String, default: "data")
    parser.add_argument("--phase", help: "Subdirectory in seq_path.", type: String, default: "train")
    parser.add_argument("--max_age", help: "Maximum number of frames to keep alive a track without associated detections.", type: Int, default: 1)
    parser.add_argument("--min_hits", help: "Minimum number of associated detections before track is initialised.", type: Int, default: 3)
    parser.add_argument("--iou_threshold", help: "Minimum IOU for match.", type: Float, default: 0.3)
    let args = parser.parse_args()
    return args
}

if #available(iOS 10.0, *) {
    let args = parse_args()
    let display = args.display
    let phase = args.phase
    var total_time: Float = 0.0
    var total_frames = 0
    var colours = [[Float]](repeating: [Float](repeating: 0, count: 3), count: 32)
    
    if display {
        if !FileManager.default.fileExists(atPath: "mot_benchmark") {
            print("\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/")
            exit(0)
        }
    }
    
    let plt = Plot()
    let fig = plt.figure()
    let ax1 = fig.add_subplot(111, aspect: "equal")
    
    if !FileManager.default.fileExists(atPath: "output") {
        try FileManager.default.createDirectory(atPath: "output", withIntermediateDirectories: true, attributes: nil)
    }
    
    let pattern = "\(args.seq_path)/\(phase)/*/det/det.txt"
    let seq_dets_fn = glob.glob(pattern)
    
    for seq_dets_fn in seq_dets_fn {
        let mot_tracker = Sort(max_age: args.max_age, min_hits: args.min_hits, iou_threshold: args.iou_threshold)
        let seq_dets = np.loadtxt(seq_dets_fn, delimiter: ",")
        let seq = seq_dets_fn[pattern.find("*")...].split(separator: "/")[0]
        
        let out_file = FileHandle(forWritingAtPath: "output/\(seq).txt")
        out_file?.truncateFile(atOffset: 0)
        
        print("Processing \(seq).")
        
        for frame in 1...Int(seq_dets[:,0].max()!) {
            var dets = seq_dets[seq_dets[:, 0]==Float(frame), 2...7]
            dets[:, 2...3] += dets[:, 0...1]
            total_frames += 1
            
            if display {
                let fn = "mot_benchmark/\(phase)/\(seq)/img1/\(String(format: "%06d.jpg", frame)))"
                let im = io.imread(fn)
                ax1.imshow(im)
                plt.title("\(seq) Tracked Targets")
            }
            
            let start_time = CFAbsoluteTimeGetCurrent()
            let trackers = mot_tracker.update(dets: dets)
            let cycle_time = CFAbsoluteTimeGetCurrent() - start_time
            total_time += Float(cycle_time)
            
            for d in trackers {
                let line = "\(frame),\(d[4]),\(d[0]),\(d[1]),\(d[2]-d[0]),\(d[3]-d[1]),1,-1,-1,-1\n"
                out_file?.write(line.data(using: .utf8)!)
                
                if display {
                    let d = d.map { Int($0) }
                    ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                }
            }
            
            if display {
                fig.canvas.flush_events()
                plt.draw()
                ax1.cla()
            }
        }
    }
    
    print("Total Tracking took: \(total_time) seconds for \(total_frames) frames or \(total_frames / total_time) FPS")
    
    if display {
        print("Note: to get real runtime results run without the option: --display")
    }
} else {
    print("This code requires iOS 10.0 or later.")
}


