//
//  HungarianAlgorithm.swift
//  ObjectDetection
//
//  Created by 신유현 on 11/16/23.
//

import Foundation

class HungarianAlgorithm : NSObject{
    var matrix: [[Double]]
    var starredZeros: [[Int]]
    var primedZeros: [[Int]]
    var coveredColumns: [Int]
    var coveredRows: [Int]

    init(inputMatrix: [[Double]]) {
        matrix = inputMatrix
        let n = matrix.count
        let m = matrix[0].count

        // Pad the matrix if necessary
        if n < m {
            matrix = matrix + Array(repeating: Array(repeating: 0.0, count: m - n), count: n)
        } else if m < n {
            matrix = matrix.map { row in
                row + Array(repeating: 0.0, count: n - row.count)
            }
        }

        starredZeros = Array(repeating: Array(repeating: 0, count: matrix[0].count), count: matrix.count)
        primedZeros = Array(repeating: Array(repeating: 0, count: matrix[0].count), count: matrix.count)
        coveredColumns = Array(repeating: 0, count: matrix[0].count)
        coveredRows = Array(repeating: 0, count: matrix.count)
    }

    func solve() -> [Int] {
        subtractRowMinima()
        subtractColumnMinima()

        while true {
            zeroStarredZeros()

            var uncoveredColumn = -1
            for col in 0..<matrix[0].count {
                if !isCoveredColumn(col: col) {
                    uncoveredColumn = col
                    break
                }
            }

            if uncoveredColumn == -1 {
                break  // All columns are covered
            }

            findStarredZeroInColumn(col: uncoveredColumn)
            while !starredZeros[uncoveredColumn].isEmpty {
                let col = starredZeros[uncoveredColumn].removeLast()
                findPrimedZeroInRow(row: col)
                findStarredZeroInColumn(col: uncoveredColumn)
            }
        }

        var assignment = Array(repeating: -1, count: matrix.count)
        for col in 0..<matrix[0].count {
            for row in 0..<matrix.count {
                if primedZeros[row][col] == 1 {
                    assignment[row] = col
                }
            }
        }

        return assignment
    }

    func subtractRowMinima() {
        for row in 0..<matrix.count {
            let minVal = matrix[row].min() ?? 0.0
            matrix[row] = matrix[row].map { $0 - minVal }
        }
    }

    func subtractColumnMinima() {
        for col in 0..<matrix[0].count {
            var minVal = Double.infinity
            for row in 0..<matrix.count {
                minVal = min(minVal, matrix[row][col])
            }
            for row in 0..<matrix.count {
                matrix[row][col] -= minVal
            }
        }
    }

    func zeroStarredZeros() {
        starredZeros = Array(repeating: Array(repeating: 0, count: matrix[0].count), count: matrix.count)
        primedZeros = Array(repeating: Array(repeating: 0, count: matrix[0].count), count: matrix.count)
        coveredColumns = Array(repeating: 0, count: matrix[0].count)
        coveredRows = Array(repeating: 0, count: matrix.count)

        for row in 0..<matrix.count {
            for col in 0..<matrix[row].count {
                if matrix[row][col] == 0 && !isCoveredColumn(col: col) && !isCoveredRow(row: row) {
                    starredZeros[row][col] = 1
                    coveredColumns[col] = 1
                    coveredRows[row] = 1
                }
            }
        }
    }

    func findStarredZeroInColumn(col: Int) {
        var row = -1
        for r in 0..<matrix.count {
            if starredZeros[r][col] == 1 {
                row = r
                break
            }
        }

        if row != -1 {
            starredZeros[row][col] = 2
        }
    }

    func findPrimedZeroInRow(row: Int) {
        var col = -1
        for c in 0..<matrix[row].count {
            if primedZeros[row][c] == 1 {
                col = c
                break
            }
        }

        if col != -1 {
            primedZeros[row][col] = 2
        }
    }

    func isCoveredColumn(col: Int) -> Bool {
        return coveredColumns[col] == 1
    }

    func isCoveredRow(row: Int) -> Bool {
        return coveredRows[row] == 1
    }
}

