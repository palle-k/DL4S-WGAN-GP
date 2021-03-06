# Deep Convolutional WGAN-GP in DL4S

This package contains an implementation of a Deep Convolutional Wasserstein-GAN with gradient penalty ([Gulrajani et al.](https://arxiv.org/abs/1704.00028)) in [DL4S](https://github.com/palle-k/DL4S).

A Wasserstein-GAN replaces the discriminator with a critic, which can output values in the range (-infinity, infinity) instead of (0, 1).
The critic is required to be 1-Lipschitz, which can either be achieved by weight clipping or using a gradient penalty as part of the loss term (the latter of which is more stable).
This implementation therefore focusses on the gradient penalty variant.

Furthermore, it uses transposed convolutions and batch normalization in the generator as well as convolutions in the critic.

### Running the Code

Download the [MNIST training images](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz) and [MNIST training labels](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz) and uncompress the gz file into the MNIST directory in this repository.

Navigate into the root directory of this repository and run using:

```bash
# Without MKL
swift run -c release
# With MKL (on Linux)
swift run -c release \
    -Xswiftc -DMKL_ENABLE \
    -Xlinker -L/opt/intel/mkl/lib/intel64 \
    -Xlinker -L/opt/intel/ipp/lib/intel64
```

Every 1000 iterations, a set of images is generated and written to ./generated/

### Computing Gradients of Gradients

As the loss term includes a gradient, it is necessary to capture the compute graph of the backpropagation operation.

```swift
let generated = optimGen.model(noise)
let eps = Tensor<Float, CPU>(Float.random(in: 0 ... 1))
let mixed = real * eps + generated * (1 - eps)

let criticScore = optimCritic.model(mixed).reduceMean()

// Compute the gradient of the critic score wrt. the input of the critic.
// By setting the retainBackwardsGraph flag to true, the compute graph of the backpropagation is captured.
// Otherwise, the gradient penalty would not be differentiable wrt. the critic parameters.
let grad = criticScore.gradients(of: [mixed], retainBackwardsGraph: true)[0]
let delta = sqrt((grad * grad).reduceSum(along: [1])) - 1

// Compute a loss that constrains the magnitude of the gradient to 1. 
// gradientPenaltyLoss can be used as a regular loss term.
let gradientPenaltyLoss = (delta * delta).reduceMean()

```
