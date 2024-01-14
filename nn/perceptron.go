package nn

import (
	"math"
	"math/rand"
)

type ActivationFunction func(float64) float64

type Perceptron struct {
	weights  []float64
	bias     float64
	activate ActivationFunction
}

// NewPerceptron creates a new perceptron with random weights and bias.
func NewPerceptron(numInputs int, activate ActivationFunction) *Perceptron {
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return &Perceptron{
		weights:  weights,
		bias:     rand.Float64(),
		activate: activate,
	}
}

// FeedForward calculates the output of the perceptron for a given input.
func (p *Perceptron) FeedForward(inputs []float64) float64 {
	sum := p.bias
	for i, input := range inputs {
		sum += input * p.weights[i]
	}
	return p.activate(sum)
}

// Train trains the perceptron on a given input and target.
func (p *Perceptron) Train(inputs []float64, target float64) {
	guess := p.FeedForward(inputs)
	err := target - guess
	for i := range p.weights {
		p.weights[i] += err * inputs[i]
	}
	p.bias += err
}

// TrainAll trains the perceptron on a given set of training data.
func (p *Perceptron) TrainAll(trainingData []Data, epochs int) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			p.Train(data.Input, data.Target)
		}
	}
}

// Data is a struct that holds the input and target for a given training example.
type Data struct {
	Input  []float64
	Target float64
}

// Sigmoid is an activation function that uses the sigmoid function.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Step is an activation function that uses the step function.
func Step(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Tanh is an activation function that uses the tanh function.
func Tanh(x float64) float64 {
	return math.Tanh(x)
}

// ReLU is an activation function that uses the ReLU function.
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// SigmoidPrime is the derivative of the sigmoid function.
func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

// StepPrime is the derivative of the step function.
func StepPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// TanhPrime is the derivative of the tanh function.
func TanhPrime(x float64) float64 {
	return 1 - math.Pow(Tanh(x), 2)
}

// ReLUPrime is the derivative of the ReLU function.
func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
