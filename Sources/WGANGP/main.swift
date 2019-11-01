//
//  main.swift
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

import DL4S
import Foundation
import SwiftGD


let (originalImage, labelsCategorical) = loadMNIST(from: "./MNIST/", type: Float.self, device: CPU.self)

let images = originalImage.view(as: [-1, 1, 28, 28])
let labels = labelsCategorical.oneHotEncoded(dim: 10, type: Float.self)

print("Creating networks...")

var optimGen = Adam(model: generator, learningRate: 0.0001, beta1: 0.0, beta2: 0.9)
var optimCrit = Adam(model: critic, learningRate: 0.0001, beta1: 0.0, beta2: 0.9)


let batchSize = 128
let epochs = 20_000
var n_critic = 5
var n_gen = 1
let lambda = Tensor<Float, CPU>(10)

print("Training...")

var runningMeanCriticLoss: Float = 0
var runningMeanGeneratorLoss: Float = 0

for epoch in 1 ... epochs {
    var lastCriticDiscriminationLoss: Tensor<Float, CPU> = 0
    var lastGradientPenaltyLoss: Tensor<Float, CPU> = 0
    
    for _ in 0 ..< n_critic {
        let (real, realLabels) = Random.minibatch(from: images, labels: labels, count: batchSize)
        
        let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
        let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
            .oneHotEncoded(dim: 10, type: Float.self)
        
        let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
        
        let fakeGenerated = optimGen.model(genInputs)
        let eps = Tensor<Float, CPU>(Float.random(in: 0 ... 1))
        let mixed = real * eps + fakeGenerated * (1 - eps)
        
        let fakeDiscriminated = optimCrit.model((mixed, genLabelInput))
        let realDiscriminated = optimCrit.model((real, realLabels))
        
        let criticDiscriminationLoss = OperationGroup.capture(named: "CriticDiscriminationLoss") {
            fakeDiscriminated.reduceMean() - realDiscriminated.reduceMean()
        }
        
        let gradientPenaltyLoss = OperationGroup.capture(named: "GradientPenaltyLoss") { () -> Tensor<Float, CPU> in
            let criticInputGrads = fakeDiscriminated.gradients(of: [mixed, genLabelInput], retainBackwardsGraph: true)
            let fakeGeneratedGrad = Tensor(stacking: [criticInputGrads[0].view(as: batchSize, -1), criticInputGrads[1].view(as: batchSize, -1)], along: 1)
            let partialPenaltyTerm = (fakeGeneratedGrad * fakeGeneratedGrad).reduceSum(along: [1]).sqrt() - 1
            let gradientPenaltyLoss = lambda * (partialPenaltyTerm * partialPenaltyTerm).reduceMean()
            return gradientPenaltyLoss
        }
        
        let criticLoss = criticDiscriminationLoss + gradientPenaltyLoss
        let criticGradients = criticLoss.gradients(of: optimCrit.model.parameters)
        
        optimCrit.update(along: criticGradients)
        
        lastCriticDiscriminationLoss = criticDiscriminationLoss.detached()
        lastGradientPenaltyLoss = gradientPenaltyLoss.detached()
    }

    var lastGeneratorLoss: Tensor<Float, CPU> = 0
    
    for _ in 0 ..< n_gen {
        let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
        let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
            .oneHotEncoded(dim: 10, type: Float.self)
        
        let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
        
        let fakeGenerated = optimGen.model(genInputs)
        let fakeDiscriminated = optimCrit.model((fakeGenerated, genLabelInput))
        let generatorLoss = -fakeDiscriminated.reduceMean()
        
        lastGeneratorLoss = generatorLoss
        let generatorGradients = generatorLoss.gradients(of: optimGen.model.parameters)

        optimGen.update(along: generatorGradients)
    }
    
    runningMeanCriticLoss = 0.1 * lastCriticDiscriminationLoss.item + 0.9 * runningMeanCriticLoss
    runningMeanGeneratorLoss = 0.1 * lastGeneratorLoss.item + 0.9 * runningMeanGeneratorLoss
    
    if runningMeanCriticLoss > 50 || runningMeanGeneratorLoss < -100 {
        n_critic = 7
    } else if runningMeanCriticLoss < -50 {
        n_critic = 3
    } else {
        n_critic = 5
    }
    
    if runningMeanGeneratorLoss > 100 {
        n_gen = 3
    } else if runningMeanGeneratorLoss > 50 {
        n_gen = 2
    } else {
        n_gen = 1
    }
    
    if epoch.isMultiple(of: 10) {
        print(" [\(epoch)/\(epochs)] [ratio: \(n_critic):\(n_gen)] loss c: \(lastCriticDiscriminationLoss), gp: \(lastGradientPenaltyLoss), g: \(lastGeneratorLoss)")
    }
    
    if epoch.isMultiple(of: 1000) {
        let genNoiseInput = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 50], min: 0, max: 1)
        let genLabelInput = Tensor<Int32, CPU>(uniformlyDistributedWithShape: [batchSize], min: 0, max: 9)
            .oneHotEncoded(dim: 10, type: Float.self)
        
        let genInputs = Tensor(stacking: [genNoiseInput, genLabelInput], along: 1)
        
        let fakeGenerated = optimGen.model(genInputs).view(as: [-1, 28, 28])
        
        for i in 0 ..< 32 {
            let slice = fakeGenerated[i].permuted(to: [1, 0]).unsqueezed(at: 0)
            guard let image = Image(slice) else {
                fatalError("Could not make image.")
            }
            let destination = URL(fileURLWithPath: "./generated/gen_\(epoch)_\(i).png")
            image.write(to: destination)
        }
    }
}
