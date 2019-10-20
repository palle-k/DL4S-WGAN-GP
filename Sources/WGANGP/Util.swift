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


func loadMNIST<Element, Device>(from path: String, type: Element.Type = Element.self, device: Device.Type = Device.self) -> (Tensor<Element, Device>, Tensor<Int32, Device>) {
    let trainingData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-images.idx3-ubyte"))
    let trainingLabelData = try! Data(contentsOf: URL(fileURLWithPath: path + "train-labels.idx1-ubyte"))
    
    let trainImages = Tensor<Element, Device>(trainingData.dropFirst(16).prefix(28 * 28 * 60_000).map(Element.init)) / 256
    let trainLabels = Tensor<Int32, Device>(trainingLabelData.dropFirst(8).prefix(60_000).map(Int32.init))
    
    return (trainImages.view(as: [-1, 1, 28, 28]), trainLabels)
}
