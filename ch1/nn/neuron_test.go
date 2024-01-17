package nn

import (
	"math"
	"testing"
)

// Function to approximate floating point comparison
func approxEqual(a, b float64) bool {
	epsilon := 0.000001 // Adjust epsilon value as needed
	return math.Abs(a-b) < epsilon
}

func TestNeuron_FeedForward(t *testing.T) {
	// Create a test neuron using NewNeuron.
	neuron := Neuron{
		g: ASigmoid,
		w: []float64{0.5, 0.5, 0.5, 0.5}, // Must contain 1 more weight than inputs.
	}

	// Test case 1: inputs = [1, 2, 3]
	inputs1 := []float64{1, 2, 3}
	expectedOutput1 := 0.970688
	output1 := neuron.FeedForward(inputs1)
	if !approxEqual(output1, expectedOutput1) {
		t.Errorf("Expected output %f, but got %f", expectedOutput1, output1)
	}

	// Test case 2: inputs = [0.5, -1, 2.5]
	inputs2 := []float64{0.5, -1, 2.5}
	expectedOutput2 := 0.817574
	output2 := neuron.FeedForward(inputs2)
	if !approxEqual(output2, expectedOutput2) {
		t.Errorf("Expected output %f, but got %f", expectedOutput2, output2)
	}
	if !approxEqual(output2, expectedOutput2) {
		t.Errorf("Expected output %f, but got %f", expectedOutput2, output2)
	}

	// Test case 3: inputs = [2, 4, 6]
	inputs3 := []float64{2, 4, 6}
	expectedOutput3 := 0.998499
	output3 := neuron.FeedForward(inputs3)
	if !approxEqual(output3, expectedOutput3) {
		t.Errorf("Expected output %f, but got %f", expectedOutput3, output3)
	}

	// Add more test cases as needed...

	// Add more test cases as needed...

}

func TestNeuron_Train(t *testing.T) {
	// Create a test neuron using NewNeuron.
	neuron := Neuron{
		g: ASigmoid,
		w: []float64{0.5, 0.5, 0.5, 0.5}, // Must contain 1 more weight than inputs.
	}

	// Test case 1: inputs = [1, 2, 3], target = 0.970688, learningRate = 0.1
	inputs1 := []float64{1, 2, 3}
	target1 := 0.970688
	learningRate1 := 0.1
	expectedWeights1 := []float64{0.5000000045979396, 0.5000000045979396, 0.5000000091958791, 0.5000000137938186}
	neuron.Train(inputs1, target1, learningRate1)
	if !approxEqual(neuron.w[0], expectedWeights1[0]) || !approxEqual(neuron.w[1], expectedWeights1[1]) ||
		!approxEqual(neuron.w[2], expectedWeights1[2]) || !approxEqual(neuron.w[3], expectedWeights1[3]) {
		t.Errorf("Expected weights %v, but got %v", expectedWeights1, neuron.w)
	}

	// Test case 2: inputs = [0.5, -1, 2.5], target = 0.817574, learningRate = 0.2
	inputs2 := []float64{0.5, -1, 2.5}
	target2 := 0.817574
	learningRate2 := 0.2
	expectedWeights2 := []float64{0.49999998415837954, 0.49999999437815956, 0.5000000296354391, 0.4999999626949185}
	neuron.Train(inputs2, target2, learningRate2)
	if !approxEqual(neuron.w[0], expectedWeights2[0]) || !approxEqual(neuron.w[1], expectedWeights2[1]) ||
		!approxEqual(neuron.w[2], expectedWeights2[2]) || !approxEqual(neuron.w[3], expectedWeights2[3]) {
		t.Errorf("Expected weights %v, but got %v", expectedWeights2, neuron.w)
	}

	// Test case 3: inputs = [2, 4, 6], target = 0.998499, learningRate = 0.3
	inputs3 := []float64{2, 4, 6}
	target3 := 0.998499
	learningRate3 := 0.3
	expectedWeights3 := []float64{0.4999999949277014, 0.5000000159168033, 0.5000000727127265, 0.5000000273108497}
	neuron.Train(inputs3, target3, learningRate3)
	if !approxEqual(neuron.w[0], expectedWeights3[0]) || !approxEqual(neuron.w[1], expectedWeights3[1]) ||
		!approxEqual(neuron.w[2], expectedWeights3[2]) || !approxEqual(neuron.w[3], expectedWeights3[3]) {
		t.Errorf("Expected weights %v, but got %v", expectedWeights3, neuron.w)
	}
}
