package nn

// Neuron holds the weights and activation function for a single neuron.
type Neuron Layer

// NewNeuron creates a new perceptron with random weights and bias.
func NewNeuron(numInputs int, act ActivationType) *Neuron {
	return (*Neuron)(NewLayer(numInputs, 1, act))
}

// FeedForward calculates the output of the perceptron for a given input.
// The input must have an extra 1 at the end for the virtual bias input.
func (p *Neuron) FeedForward(inputs []float64) float64 {
	return (*Layer)(p).FeedForward(inputs)[0]
}

// Train trains the perceptron on a given input and target.
// The input must have an extra 1 at the end for the virtual bias input.
func (p *Neuron) Train(inputs []float64, target float64, learningRate float64) {
	(*Layer)(p).Train(inputs, []float64{target}, learningRate)
}

// TrainAll trains the perceptron on a given set of training data.
func (p *Neuron) TrainAll(trainingData []Data, epochs int, learningRate float64) {
	(*Layer)(p).TrainAll(trainingData, epochs, learningRate)
}
