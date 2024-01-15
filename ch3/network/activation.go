// File to contain activation functions.
package network

import "math"

type ActivationType int

const (
	AStep ActivationType = iota
	ASigmoid
	AReLU
	ATanh
)

var (
	ActivationFuncs = map[ActivationType]ActivationFunction{
		AStep:    Step,
		ASigmoid: Sigmoid,
		AReLU:    ReLU,
		ATanh:    Tanh,
	}
	ActivationPrimes = map[ActivationType]ActivationFunction{
		AStep:    StepPrime,
		ASigmoid: SigmoidPrime,
		AReLU:    ReLUPrime,
		ATanh:    TanhPrime,
	}
)

// Step is an activation function that uses the step function.
func Step(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// StepPrime is the derivative of the step function.
func StepPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid is an activation function that uses the sigmoid function.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidPrime is the derivative of the sigmoid function.
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

// ReLU is an activation function that uses the ReLU function.
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUPrime is the derivative of the ReLU function.
func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh is an activation function that uses the tanh function.
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// TanhPrime is the derivative of the tanh function.
func TanhPrime(x float64) float64 {
	return 1 - math.Pow(Tanh(x), 2)
}
