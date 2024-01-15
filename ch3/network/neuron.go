package network

import (
	"math/rand"
)

type ActivationFunction func(float64) float64

type Neuron []float64 // Last one is bias.

// NewNeuron creates a new perceptron with random weights and bias.
func NewNeuron(numInputs int) Neuron {
	weights := make([]float64, numInputs+1)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return Neuron(weights)
}

// FeedForward calculates the output of the perceptron for a given input.
// The input must have an extra 1 at the end for the virtual bias input.
func (p Neuron) FeedForward(inputs []float64, activate ActivationType) float64 {
	acFunc := ActivationFuncs[activate]
	return acFunc(Dot(inputs, p[1:]) + p[0])
}

// Train trains the perceptron on a given input and target.
// The input must have an extra 1 at the end for the virtual bias input.
func (p Neuron) Train(inputs []float64, target float64, activate ActivationType, learningRate float64) {
	output := p.FeedForward(inputs, activate)
	delta := (target - output) * ActivationPrimes[activate](output)
	for i := range inputs {
		p[i+1] += learningRate * delta * inputs[i]
	}
	p[0] += learningRate * delta
}

// TrainAll trains the perceptron on a given set of training data.
func (p Neuron) TrainAll(trainingData []Data, epochs int, activate ActivationType, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			p.Train(data.Input, data.Target, activate, learningRate)
		}
	}
}
