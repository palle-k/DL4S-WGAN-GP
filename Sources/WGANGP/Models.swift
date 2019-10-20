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

var generator = Sequential {
    Dense<Float, CPU>(inputSize: 50 + 10, outputSize: 200)
    BatchNorm<Float, CPU>(inputSize: [200])
    LeakyRelu<Float, CPU>(leakage: 0.2)
    
    Dense<Float, CPU>(inputSize: 200, outputSize: 800)
    BatchNorm<Float, CPU>(inputSize: [800])
    LeakyRelu<Float, CPU>(leakage: 0.2)
    
    Dense<Float, CPU>(inputSize: 800, outputSize: 28 * 28)
    Sigmoid<Float, CPU>()
}

var critic = Sequential {
    Dense<Float, CPU>(inputSize: 28 * 28 + 10, outputSize: 400)
    Relu<Float, CPU>()
    
    Dense<Float, CPU>(inputSize: 400, outputSize: 100)
    Relu<Float, CPU>()
    
    Dense<Float, CPU>(inputSize: 100, outputSize: 1)
}
