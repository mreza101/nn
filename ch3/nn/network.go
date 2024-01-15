package nn

// Network is a muli-layer network of perceptrons.
type Network struct {
	layers  []*Layer    // The first layer is the output layer, the last layer is the input layer
	outputs [][]float64 // The outputs of each layer
}

// NewNetwork creates a new empty network.
func NewNetwork() *Network {
	return &Network{}
}

// AddLayer adds a new layer to the network. The first layer added is the output layer, the last layer added is the input layer.
// numUnnumOutputsts is the number of neurons, or output units, in the layer.
// numInputs is the number of inputs to the layer. Also the number of outputs of the previous layer.
// act is the activation function of the layer.
func (n *Network) AddLayer(numOutputs, numInputs int, act ActivationType) {
	n.layers = append(n.layers, NewLayer(numOutputs, numInputs, act))
}

// FeedForward calculates the output of the network for a given input.
func (n *Network) FeedForward(inputs []float64) []float64 {
	n.outputs = make([][]float64, len(n.layers))
	for i := len(n.layers) - 1; i >= 0; i-- {
		n.outputs[i] = n.layers[i].FeedForward(inputs)
		inputs = n.outputs[i]
	}
	return n.outputs[0] // First layer is the output layer.
}

// Train trains the network on a given input and target using the backpropagation algorithm.
func (n *Network) Train(inputs, targets []float64, learningRate float64) {
	outs := n.FeedForward(inputs)
	n.layers[0].Train(outs, targets, learningRate)
	// TODO: fix this. I need to keep the delta of each layer, starting from the output layer, so that I can calculate the delta of the previous layer.
}

// TrainAll trains the network on a given set of training data.
func (n *Network) TrainAll(trainingData []Data, epochs int, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			n.Train(data.Input, data.Targets, learningRate)
		}
	}
}

// Test calculates the accuracy of the network for a given set of test data.
func (n *Network) Test(testData []Data) float64 {
	numCorrect := 0
	for _, data := range testData {
		outputs := n.FeedForward(data.Input)
		if maxIndex(outputs) == maxIndex(data.Targets) {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(len(testData))
}

func maxIndex(xs []float64) int {
	maxIndex := 0
	for i := range xs {
		if xs[i] > xs[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}
