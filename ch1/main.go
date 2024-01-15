package main

import (
	"flag"
	"fmt"

	nn "github.com/mreza101/gonn/ch1/nn"
)

// trainingData for a logical OR
var trainingData = []nn.Data{
	{Input: []float64{0, 0}, Target: 0},
	{Input: []float64{0, 1}, Target: 1},
	{Input: []float64{1, 0}, Target: 1},
	{Input: []float64{1, 1}, Target: 1},
}

var epochs = flag.Int("epochs", 10, "Number of epochs to train for")
var learningRate = flag.Float64("learning-rate", 0.1, "Learning rate to use")

func main() {
	// Create a new perceptron.
	p := nn.NewNeuron(2, nn.ASigmoid)

	// Train the perceptron.
	p.TrainAll(trainingData, *epochs, *learningRate)

	// Test the perceptron.
	fmt.Printf("0,0 = %f\n", p.FeedForward([]float64{0, 0}))
	fmt.Printf("0,1 = %f\n", p.FeedForward([]float64{0, 1}))
	fmt.Printf("1,0 = %f\n", p.FeedForward([]float64{1, 0}))
	fmt.Printf("1,1 = %f\n", p.FeedForward([]float64{1, 1}))
}
