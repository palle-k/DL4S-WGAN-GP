//
//  Util.swift
//  WGANGP
//
//  Created by Palle Klewitz on 20.10.19.
//  Copyright (c) 2019 Palle Klewitz
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

import Foundation
import DL4S
import SwiftGD


extension DispatchQueue {
    static func concurrentPerform<Result>(units: Int, workers: Int, task: @escaping (Int) -> Result) -> [Result] {
        let sema = DispatchSemaphore(value: 0)
        let tasksPerWorker = units / workers
        
        var results = [Result?](repeating: nil, count: units)
        
        for workerID in 0 ..< workers {
            let taskRange: CountableRange<Int>
            if workerID == workers - 1 {
                taskRange = tasksPerWorker * workerID ..< units
            } else {
                taskRange = tasksPerWorker * workerID ..< tasksPerWorker * (workerID + 1)
            }
            
            DispatchQueue.global().async {
                for unitID in taskRange {
                    results[unitID] = task(unitID)
                }
                
                sema.signal()
            }
        }
        
        for _ in 0 ..< workers {
            sema.wait()
        }
        
        return results.compactMap {$0}
    }
}


func loadMNIST<Element, Device>(from path: String, type: Element.Type = Element.self, device: Device.Type = Device.self) -> (Tensor<Element, Device>, Tensor<Int32, Device>) {
    let trainingData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-images.idx3-ubyte"))
    let trainingLabelData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-labels.idx1-ubyte"))
    
    let trainImages = Tensor<Element, Device>(trainingData.dropFirst(16).prefix(28 * 28 * 60_000).map(Element.init)) / 256
    let trainLabels = Tensor<Int32, Device>(trainingLabelData.dropFirst(8).prefix(60_000).map(Int32.init))
    
    return (trainImages.view(as: [-1, 1, 28, 28]), trainLabels)
}


extension Image {
    convenience init?<E: NumericType, D: DeviceType>(_ tensor: Tensor<E, D>) {
        precondition(2 ... 3 ~= tensor.dim, "Tensor must have 2 or 3 dimensions.")
        let t: Tensor<E, D>
        if tensor.dim == 3 {
            t = tensor.detached()
        } else {
            t = tensor.detached().unsqueezed(at: 0)
        }

        let (width, height) = (t.shape[2], t.shape[1])
        self.init(width: width, height: height)

        for y in 0 ..< height {
            for x in 0 ..< width {
                let color: Color
                let slice = t[nil, y, x]
                if slice.count == 1 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[0].item.doubleValue, blue: slice[0].item.doubleValue, alpha: 1)
                } else if slice.count == 3 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[1].item.doubleValue, blue: slice[2].item.doubleValue, alpha: 1)
                } else if slice.count == 4 {
                    color = Color(red: slice[0].item.doubleValue, green: slice[1].item.doubleValue, blue: slice[2].item.doubleValue, alpha: slice[3].item.doubleValue)
                } else {
                    fatalError("Unsupported format. Tensor must have shape [height, width], [1, height, width], [3, height, width] or [4, height, width]")
                }
                self.set(pixel: Point(x: x, y: y), to: color)
            }
        }
    }
}

func unzip<U, V, S: Sequence>(_ sequence: S) -> ([U], [V]) where S.Element == (U, V) {
    var u: [U] = []
    var v: [V] = []
    u.reserveCapacity(sequence.underestimatedCount)
    v.reserveCapacity(sequence.underestimatedCount)
    
    for (ue, ve) in sequence {
        u.append(ue)
        v.append(ve)
    }
    
    return (u, v)
}
