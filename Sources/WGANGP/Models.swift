//
//  Models.swift
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

let generator = Sequential {
    Dense<Float, CPU>(inputSize: 50 + 10, outputSize: 512)
    BatchNorm<Float, CPU>(inputSize: [512])
    LeakyRelu<Float, CPU>(leakage: 0.2)
    
    Reshape<Float, CPU>(outputShape: [8, 8, 8])
    
    TransposedConvolution2D<Float, CPU>(inputChannels: 8, outputChannels: 6, kernelSize: (3, 3), padding: 1, stride: 2)
    BatchNorm<Float, CPU>(inputSize: [6, 15, 15])
    LeakyRelu<Float, CPU>(leakage: 0.2)
    
    TransposedConvolution2D<Float, CPU>(inputChannels: 6, outputChannels: 3, kernelSize: (3, 3), padding: 1, stride: 2)
    BatchNorm<Float, CPU>(inputSize: [3, 29, 29])
    LeakyRelu<Float, CPU>(leakage: 0.2)
    
    Convolution2D<Float, CPU>(inputChannels: 3, outputChannels: 1, kernelSize: (2, 2), padding: 0, stride: 1)
    Sigmoid<Float, CPU>()
}

struct Critic<Element: RandomizableType, Device: DeviceType>: LayerType {
    var parameters: [Tensor<Element, Device>] {
        convolutions.parameters + classifier.parameters
    }
    var parameterPaths: [WritableKeyPath<Self, Tensor<Element, Device>>] {
        Array([
            convolutions.parameterPaths.map((\Self.convolutions).appending(path:)),
            classifier.parameterPaths.map((\Self.classifier).appending(path:))
        ].joined())
    }
    
    var convolutions = Sequential {
        Convolution2D<Element, Device>(inputChannels: 1, outputChannels: 6, kernelSize: (3, 3))  // 28x28
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 14x14
        
        Convolution2D<Element, Device>(inputChannels: 6, outputChannels: 12, kernelSize: (3, 3), padding: 2)  // 16x16
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 8x8
        
        Convolution2D<Element, Device>(inputChannels: 12, outputChannels: 16, kernelSize: (3, 3))  // 8x8
        Relu<Element, Device>()
        
        MaxPool2D<Element, Device>(windowSize: 2, stride: 2, padding: 0) // 4x4
        Flatten<Element, Device>() // 256
    }
    
    var classifier = Sequential {
        Concat<Element, Device>()
        
        Dense<Element, Device>(inputSize: 256 + 10, outputSize: 128)
        Relu<Element, Device>()
        
        Dense<Element, Device>(inputSize: 128, outputSize: 1)
    }
    
    init() {
        convolutions.tag = "Convolutions"
        classifier.tag = "Classifier"
    }
    
    func callAsFunction(_ inputs: (Tensor<Element, Device>, Tensor<Element, Device>)) -> Tensor<Element, Device> {
        OperationGroup.capture(named: "Critic") {
            let conv = convolutions(inputs.0)
            return classifier([conv, inputs.1])
        }
    }
}

let critic = Critic<Float, CPU>()
