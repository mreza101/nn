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
// Assumptions:
//  1. The length of targets is equal to the number of output units in the output layer, which is layer 0.
//  2. The length of inputs is equal to the number of input units in the input layer, which is the last layer.
func (n *Network) Train(inputs, targets []float64, learningRate float64) {
	// Step 1 - Feed forward and compute the output of each layer.
	n.FeedForward(inputs) // We ignore the output because we already have it in n.outputs.

	// Step 2 - Calculate the deltas of all units of all layers. Starting with the output layer.
	deltas := make([][]float64, len(n.layers)) // deltas[i][j] is the delta of the jth unit of the ith layer.

	//   Step 2.1 - Calculate the deltas of the output layer.
	acPrime := ActivationPrimes[n.layers[0].g] // Derivative of the activation function of the output layer.
	output := n.outputs[0]                     // Output of the output layer.
	deltas[0] = make([]float64, len(output))   // Deltas of the output layer.
	for i := range output {                    // Iterate over all output units.
		deltas[0][i] = (targets[i] - output[i]) * acPrime(output[i])
	}

	//   Step 2.2 - Calculate the deltas of the hidden layers.
	for i := 1; i < len(n.layers); i++ { // Iterate over all hidden layers.
		acPrime = ActivationPrimes[n.layers[i].g] // Derivative of the activation function of the hidden layer.
		output = n.outputs[i]                     // Output of the hidden layer.
		deltas[i] = make([]float64, len(output))  // Deltas of the hidden layer.
		for j := range output {                   // Iterate over all hidden units.
			// Calculate the error for each hidden neuron by looking at the errors of the output neurons
			deltas[i][j] = 0
			for k := range n.layers[i-1].w { // Iterate over all output units of the layer above.
				deltas[i][j] += deltas[i-1][k] * n.layers[i-1].w[k][j+1] // TODO: Verify this.
			}
			deltas[i][j] *= acPrime(output[j])
		}
	}

	// Step 3 - Update the weights of all layers.
	for i := range n.layers { // Iterate over all layers.
		for j := range n.layers[i].w { // Iterate over all units of the layer.
			delta := deltas[i][j]
			factor := learningRate * delta
			n.layers[i].w[j][0] += factor // Adjust bias.
			prevInput := n.layers[i].w[j]
			for k := range prevInput {
				n.layers[i].w[j][k+1] += factor * prevInput[k]
			}
		}
	}
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
