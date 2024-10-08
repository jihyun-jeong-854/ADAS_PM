// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import UIKit

class ObjectDetector {
    lazy var detectmodule: InferenceModule = {
        if let filePath = Bundle.main.path(forResource: "exp13_best_v5.torchscript", ofType: "ptl"),
            let module = InferenceModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Failed to load yolo model!")
        }
    }()
   
    lazy var classes: [String] = {
        if let filePath = Bundle.main.path(forResource: "classes", ofType: "txt"),
            let classes = try? String(contentsOfFile: filePath) {
            return classes.components(separatedBy: .newlines)
        } else {
            fatalError("classes file was not found.")
        }
    }()
}
