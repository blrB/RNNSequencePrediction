# Artificial Neural Networks (ANN) Sequence Prediction

Recurrent Neural Network (RNN) for laboratory work.

## Library

<a href="http://arma.sourceforge.net/docs.html/">Armadillo C++ linear algebra library</a>

for Ubuntu:

```sh
sudo apt-get install libarmadillo-dev libarmadillo6
```
Before installing Armadillo, it's recommended to install LAPACK, BLAS and ATLAS

## Input:

p – number of neuron;

m – number of neuron in hidden layer;

e – error degree;

α – learning step;

## Example for Fibonacci number

Input:

n = 2; m = 4; e = 0.0048; a = 0.000001; predict = 10;

0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610

Output:

987, 1597, 2584, 4181.01, 6765.01, 10946, 17711.1, 28657, 46368.1, 75025.1


