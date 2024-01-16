package nn

import (
	"math/rand"
)

// Data is a struct that holds the input and targets for a given training example.
type Data struct {
	Input   []float64
	Targets []float64
}

// Neuron holds the weights and activation function for a single neuron.
type Layer struct {
	w [][]float64    // Each row correspond to one output unit for this layer. Each row has one extra element (the first one) for bias.
	g ActivationType // Type of activation function for all the output units of this layer.
}

// NewLayer creates a new perceptron with random weights and bias.
func NewLayer(numInputs, numOutputs int, act ActivationType) *Layer {
	weights := make([][]float64, numOutputs)
	for i := range weights {
		weights[i] = make([]float64, numInputs+1)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}

	}
	return &Layer{w: weights, g: act}
}

func dot(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

// FeedForward calculates the output of the perceptron for a given input.
func (p *Layer) FeedForward(inputs []float64) []float64 {
	acFunc := ActivationFuncs[p.g]

	outputs := make([]float64, len(p.w))
	for i := range outputs {
		outputs[i] = acFunc(dot(inputs, p.w[i][1:]) + p.w[i][0])
	}
	return outputs
}

// Train trains a layer of perceptrons on the given inputs and targets.
func (p *Layer) Train(inputs []float64, targets []float64, learningRate float64) {
	acPrime := ActivationPrimes[p.g]
	output := p.FeedForward(inputs)
	for i := range p.w { // Iterate over all output units.
		delta := (targets[i] - output[i]) * acPrime(output[i])
		factor := learningRate * delta
		p.w[i][0] += factor // Adjust bias.
		for j := range inputs {
			p.w[i][j+1] += factor * inputs[j]
		}
	}
}

// TrainAll trains the perceptron on a given set of training data.
func (p *Layer) TrainAll(trainingData []Data, epochs int, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			p.Train(data.Input, data.Targets, learningRate)
		}
	}
}
