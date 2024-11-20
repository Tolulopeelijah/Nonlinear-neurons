# My Quadratic Neuron Experiments

This is a personal project where I'm experimenting with using quadratic transformations in neural networks to solve the XOR problem. The idea is to replace traditional linear transformations in neurons with quadratic equations and see if it helps capture non-linear patterns more efficiently.

---

## Why I'm Doing This
The XOR problem is a classic example that demonstrates the limitations of simple perceptrons. While it's easy to solve with two-layer neural networks, I started wondering if we could solve it using just one layer by tweaking the neuron itself to include quadratic terms.

This is an experiment to explore:
1. How quadratic terms like \(x^2\) and \(xy\) can help solve XOR.
2. Whether this approach might reduce the number of layers needed in a network.

---

## What I'm Trying
In traditional neurons, we have something like:
\[z = Wx + b\]

In my quadratic neurons, I’m adding terms like:
\[z = w_1x_1^2 + w_2x_2^2 + w_3x_1x_2 + w_4x_1 + w_5x_2 + b\]

This introduces quadratic and interaction terms directly into the neuron, which might make it easier to form non-linear decision boundaries.

---

## Current Status
- **Setup:** I've written a basic PyTorch model to test the idea.  
- **Goal:** To see if a single-layer network with quadratic neurons can solve the XOR problem.  

---

## How to Run My Code
If you’re curious to try this (or future me wants to revisit this project), here’s how to get started:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MyQuadraticNeuronExperiments.git
   cd MyQuadraticNeuronExperiments
