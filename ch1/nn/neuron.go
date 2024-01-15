package nn

import (
	"math/rand"
)

// Data is a struct that holds the input and target for a given training example.
type Data struct {
	Input  []float64
	Target float64
}

type ActivationFunction func(float64) float64

type Neuron struct {
	w []float64      // Last one is bias.
	g ActivationType // Type of activation function.
}

// NewNeuron creates a new perceptron with random weights and bias.
func NewNeuron(numInputs int, act ActivationType) *Neuron {
	weights := make([]float64, numInputs+1)
	for i := range weights {
		weights[i] = rand.Float64()
	}
	return &Neuron{w: weights, g: act}
}

func dot(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

// FeedForward calculates the output of the perceptron for a given input.
// The input must have an extra 1 at the end for the virtual bias input.
func (p *Neuron) FeedForward(inputs []float64) float64 {
	acFunc := ActivationFuncs[p.g]
	return acFunc(dot(inputs, p.w[1:]) + p.w[0])
}

// Train trains the perceptron on a given input and target.
// The input must have an extra 1 at the end for the virtual bias input.
func (p *Neuron) Train(inputs []float64, target float64, learningRate float64) {
	output := p.FeedForward(inputs)
	delta := (target - output) * ActivationPrimes[p.g](output)
	factor := learningRate * delta
	p.w[0] += factor // Adjust bias.
	for i := range inputs {
		p.w[i+1] += factor * inputs[i]
	}
}

// TrainAll trains the perceptron on a given set of training data.
func (p *Neuron) TrainAll(trainingData []Data, epochs int, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			p.Train(data.Input, data.Target, learningRate)
		}
	}
}
