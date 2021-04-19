//
//  main.swift
//  WGANGP
//
//  Created by Palle Klewitz on 20.10.19.
//  Copyright (c) 2019 - 2020 Palle Klewitz
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
import DL4STensorboard
import Foundation
import SwiftGD


let (originalImage, labelsCategorical) = loadMNIST(from: "./MNIST/", type: Float.self, device: CPU.self)

let images = originalImage.view(as: [-1, 1, 28, 28])
let labels = labelsCategorical.oneHotEncoded(dim: 10, type: Float.self)

print("Creating networks...")

var optimGen = Adam(model: generatorV2, learningRate: 0.001, beta1: 0.0, beta2: 0.9)
var optimCrit = Adam(model: criticV2, learningRate: 0.001, beta1: 0.0, beta2: 0.9)

func writeModels(_ epoch: Int) throws {
    let encoder = JSONEncoder()
    encoder.dataEncodingStrategy = .base64
    
    let genData = try encoder.encode(optimGen)
    try genData.write(to: URL(fileURLWithPath: "./generator_v2.\(epoch).json"))
    
    let critData = try encoder.encode(optimCrit)
    try critData.write(to: URL(fileURLWithPath: "./critic_v2.\(epoch).json"))
}

let totalBatchSize = 256
let samplingBatchSize = 64
let epochs = 50_000
var n_critic = 5
var n_gen = 1
let lambda: Tensor<Float, CPU> = 10
let workers = 8

print("Training...")

let writer = try TensorboardWriter(logDirectory: URL(fileURLWithPath: "./logs"), runName: "bs256_v2_5v1_l10_run3")

for epoch in 1 ... epochs {
    var lastCriticDiscriminationLoss: Tensor<Float, CPU> = 0
    var lastGradientPenaltyLoss: Tensor<Float, CPU> = 0
    
    // for every generator optimization step, the discriminator can be updated multiple times.
    // for a wgan, the critic can be trained towards optimality without gradients becoming unstable
    for _ in 0 ..< n_critic {
        // split batch over multiple threads
        let results = DispatchQueue.concurrentPerform(units: workers, workers: workers) { _ -> ((Tensor<Float, CPU>, Tensor<Float, CPU>), [Tensor<Float, CPU>]) in
            let batchSize = totalBatchSize / workers
            
            // sample from real distribution
            let (real, _) = Random.minibatch(from: images, labels: labels, count: batchSize)
            
            // sample from generated distribution
            let genInputs = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 256], min: 0, max: 1)
            let fakeGenerated = optimGen.model(genInputs)
            
            // interpolate between generated and real sample
            let eps = Tensor<Float, CPU>(Float.random(in: 0 ... 1))
            var mixed = (real * eps + fakeGenerated * (1 - eps)).detached()
            // we want to compute the gradient of the prediction wrt. mixed, so mixed requires a gradient
            mixed.requiresGradient = true
            
            let fakeDiscriminated = optimCrit.model(fakeGenerated)
            let realDiscriminated = optimCrit.model(real)
            let mixedDiscriminated = optimCrit.model(mixed)
            
            let criticDiscriminationLoss = OperationGroup.capture(named: "Wasserstein Loss") {
                fakeDiscriminated.reduceMean() - realDiscriminated.reduceMean()
            }
            
            let gradientPenaltyLoss = OperationGroup.capture(named: "Gradient Penalty") { () -> Tensor<Float, CPU> in
                // compute gradient of prediction wrt. mixed.
                // retain the backward graph, so that we can compute the gradient of the model weights wrt. an expression derived from a gradient.
                let mixedGrads = mixedDiscriminated.gradients(of: [mixed], retainBackwardsGraph: true)[0]
                    .view(as: batchSize, -1)
                
                // model should be somewhat linear, so gradient of mixed must be close to 1.
                let partialPenaltyTerm = (mixedGrads * mixedGrads).reduceSum(along: [1]).sqrt() - 1
                let gradientPenaltyLoss = (partialPenaltyTerm * partialPenaltyTerm).reduceMean()
                return gradientPenaltyLoss
            }
            
            let criticLoss = criticDiscriminationLoss + lambda * gradientPenaltyLoss
            
            return (
                (criticDiscriminationLoss.detached(), gradientPenaltyLoss.detached()),
                criticLoss.gradients(of: optimCrit.model.parameters)
            )
        }
        
        // combine gradients from all threads.
        let (losses, criticGradsBatch) = unzip(results)
        let criticGradients = criticGradsBatch.dropFirst().reduce(criticGradsBatch.first!) { acc, grads in
            zip(acc, grads).map(+)
        }.map { grad in
            grad / Tensor(Float(workers))
        }
        
        optimCrit.update(along: criticGradients)
        
        let (discriminationLosses, gradientPenaltyLosses) = unzip(losses)
        lastCriticDiscriminationLoss = discriminationLosses.reduce(0, +) / Tensor(Float(workers))
        lastGradientPenaltyLoss = gradientPenaltyLosses.reduce(0, +) / Tensor(Float(workers))
    }

    var lastGeneratorLoss: Tensor<Float, CPU> = 0
    
    for _ in 0 ..< n_gen {
        // split computation over multiple threads
        let results = DispatchQueue.concurrentPerform(units: workers, workers: workers) { _ -> (Tensor<Float, CPU>, [Tensor<Float, CPU>]) in
            let batchSize = totalBatchSize / workers
            
            let genInputs = Tensor<Float, CPU>(uniformlyDistributedWithShape: [batchSize, 256], min: 0, max: 1)
            
            let fakeGenerated = optimGen.model(genInputs)
            let fakeDiscriminated = optimCrit.model(fakeGenerated)
            let generatorLoss = -fakeDiscriminated.reduceMean()
            
            let generatorGradients = generatorLoss.gradients(of: optimGen.model.parameters)
            
            return (generatorLoss, generatorGradients)
        }
        
        let (losses, grads) = unzip(results)
        
        // combine gradients from all threads
        lastGeneratorLoss = losses.reduce(0, +) / Tensor(Float(workers))
        let generatorGradients = grads.dropFirst().reduce(grads.first!) { acc, grads in
            zip(acc, grads).map(+)
        }.map { grad in
            grad / Tensor(Float(workers))
        }

        optimGen.update(along: generatorGradients)
    }
    
    try? writer.write(scalar: -lastCriticDiscriminationLoss.item, withTag: "critic/neg_loss", atStep: epoch)
    try? writer.write(scalar: lastGradientPenaltyLoss.item, withTag: "critic/gradient_penalty", atStep: epoch)
    try? writer.write(scalar: lastGeneratorLoss.item, withTag: "generator/loss", atStep: epoch)
    
    if epoch.isMultiple(of: 10) {
        print(" [\(epoch)/\(epochs)] [ratio: \(n_critic):\(n_gen)] loss c: \(lastCriticDiscriminationLoss), gp: \(lastGradientPenaltyLoss), g: \(lastGeneratorLoss)")
    }
    
    // save some samples every 100 iterations
    if epoch.isMultiple(of: 100) {
        do {
            try writeModels(epoch)
        } catch {
            print("Could not write models")
        }
        
        let genInputs = Tensor<Float, CPU>(uniformlyDistributedWithShape: [samplingBatchSize, 256], min: 0, max: 1)
        let fakeGenerated = optimGen.model(genInputs).view(as: [-1, 28, 28])
        
        for i in 0 ..< samplingBatchSize {
            let slice = fakeGenerated[i].unsqueezed(at: 0)
            try? writer.write(image: slice, withTag: "generator/output", atStep: epoch)
        }
    }
}

try writeModels(epochs)
