package network

type Layer struct {
	neurons      []Neuron
	activate     ActivationFunction
	learningRate float64
}

// NewLayer creates a new layer of perceptrons.
func NewLayer(numNeurons, numInputs int, activate ActivationFunction, learningRate float64) *Layer {
	neurons := make([]Neuron, numNeurons)
	for i := range neurons {
		neurons[i] = Neuron(make([]float64, numInputs+1))
	}
	return &Layer{
		neurons:      neurons,
		activate:     activate,
		learningRate: learningRate,
	}
}

// FeedForward calculates the output of the layer for a given input.
type Network struct {
	hiddenLayer *Layer
	outputLayer *Layer

	hiddenOutputs []float64
}

// NewNetwork creates a new neural network with random weights and bias.
func NewNetwork(numInputs, numHidden, numOutputs int, activate ActivationFunction, learningRate float64) *Network {
	hiddenLayer := NewLayer(numHidden, numInputs, activate, learningRate)
	outputLayer := NewLayer(numOutputs, numHidden, activate, learningRate)
	return &Network{
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
	}
}

// FeedForward calculates the output of the network for a given input.
func (n *Network) FeedForward(inputs []float64) []float64 {
	n.hiddenOutputs = make([]float64, len(n.hiddenLayer.neurons))
	for i, neuron := range n.hiddenLayer.neurons {
		n.hiddenOutputs[i] = neuron.FeedForward(inputs)
	}
	outputs := make([]float64, len(n.outputLayer.neurons))
	for i, neuron := range n.outputLayer.neurons {
		outputs[i] = neuron.FeedForward(n.hiddenOutputs)
	}
	return outputs
}

// Train trains the network on a given input and target using the backpropagation algorithm.
func (n *Network) Train(inputs, targets []float64) {
	outputs := n.FeedForward(inputs)
	outputErrors := make([]float64, len(n.outputLayer.neurons))
	for i := range n.outputLayer.neurons {
		outputErrors[i] = targets[i] - outputs[i]
	}
	hiddenErrors := make([]float64, len(n.hiddenLayer.neurons))
	for i := range n.hiddenLayer.neurons {
		// Calculate the error for each hidden neuron by looking at the errors of the output neurons
		hiddenErrors[i] = 0
		for j := range n.outputLayer.neurons {
			hiddenErrors[i] += n.outputLayer.neurons[j].Weights[i] * outputErrors[j]
		}
	}
}

// TrainAll trains the network on a given set of training data.
func (n *Network) TrainAll(trainingData []Data, epochs int) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			n.Train(data.Input, data.Target)
		}
	}
}

// Predict calculates the output of the network for a given input.
func (n *Network) Predict(inputs []float64) []float64 {
	return n.FeedForward(inputs)
}

// Test calculates the accuracy of the network for a given set of test data.
func (n *Network) Test(testData []Data) float64 {
	numCorrect := 0.0
	for _, data := range testData {
		outputs := n.Predict(data.Input)
		if data.Target == maxIndex(outputs) {
			numCorrect++
		}
	}
	return numCorrect / float64(len(testData))
}

// maxIndex returns the index of the largest element in a slice.
func maxIndex(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i, value := range slice {
		if value > maxValue {
			maxIndex = i
			maxValue = value
		}
	}
	return maxIndex
}
