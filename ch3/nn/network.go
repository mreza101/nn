package nn

// Network is a muli-layer network of perceptrons.
type Network struct {
	layers  []*Layer    // The first layer (layer 0) is the output layer, the last layer is the input layer
	outputs [][]float64 // Contains L+1, 0..L, entries. The outputs of each layer. The output of layer l is the input of layer l-1. outputs[L] is the input of the network.
}

// NewNetwork creates a new empty network.
func NewNetwork() *Network {
	return &Network{}
}

// AddLayer adds a new layer to the network. The first layer added is the output layer, the last layer added is the input layer.
// numUnnumOutputsts is the number of neurons, or output units, in the layer.
// numInputs is the number of inputs to the layer. Also the number of outputs of the previous layer.
// act is the activation function of the layer.
func (n *Network) AddLayer(numInputs, numOutputs int, act ActivationType) {
	n.layers = append(n.layers, NewLayer(numInputs, numOutputs, act))
}

// FeedForward calculates the output of the network for a given input.
func (n *Network) FeedForward(inputs []float64) []float64 {
	n.outputs = make([][]float64, len(n.layers)+1)
	n.outputs[len(n.layers)] = inputs // Dummy entry so that we have the inputs in the same array during backpropagation.
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
	output := n.outputs[0]                     // Output of the output layer (layer 0).
	deltas[0] = make([]float64, len(output))   // Deltas of the output layer (layer 0).
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
	for l := range n.layers { // Iterate over all layers.
		w := n.layers[l].w // Weights of the layer.
		for j := range w { // Iterate over all units of the layer.
			factor := learningRate * deltas[l][j]
			w[j][0] += factor         // Adjust bias.
			for k := range w[j][1:] { // Adjust weights of all inputs.
				w[j][k+1] += factor * n.outputs[l+1][j] // outputs[l+1] is the input to layer l.
			}
		}
	}
}

// TrainAll trains the network on a given set of training data.
func (n *Network) TrainAll(inputs, targets [][]float64, epochs int, learningRate float64) {
	for i := 0; i < epochs; i++ {
		for t := range inputs {
			n.Train(inputs[t], targets[t], learningRate)
		}
	}
}

// Test calculates the accuracy of the network for a given set of test data.
func (n *Network) Test(inputs, targets [][]float64) float64 {
	numCorrect := 0
	for t := range inputs {
		outputs := n.FeedForward(inputs[t])
		if maxIndex(outputs) == maxIndex(targets[t]) {
			numCorrect++
		}
	}
	return float64(numCorrect) / float64(len(inputs))
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
