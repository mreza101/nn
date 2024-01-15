package main

import (
	"flag"
	"fmt"

	nn "github.com/mreza101/gonn/ch3/network"
)

// Declare a training data set
var trainingData = []nn.Data{
	{Input: []float64{0, 0}, Target: 0},
	{Input: []float64{0, 1}, Target: 1},
	{Input: []float64{1, 0}, Target: 1},
	{Input: []float64{1, 1}, Target: 1},
}

// Delcare command line flag for epochs
var epochs = flag.Int("epochs", 10, "Number of epochs to train for")
var activation = flag.String("activation", "sigmoid", "Activation function to use")
var learningRate = flag.Float64("learning-rate", 0.1, "Learning rate to use")

// Main function
func main() {
	// Create a new perceptron
	p := nn.NewNeuron(2)
	// Train the perceptron
	p.TrainAll(trainingData, *epochs, nn.ASigmoid, *learningRate)
	// Test the perceptron
	fmt.Printf("0,0 = %f\n", p.FeedForward([]float64{0, 0}, nn.ASigmoid))
	fmt.Printf("0,1 = %f\n", p.FeedForward([]float64{0, 1}, nn.ASigmoid))
	fmt.Printf("1,0 = %f\n", p.FeedForward([]float64{1, 0}, nn.ASigmoid))
	fmt.Printf("1,1 = %f\n", p.FeedForward([]float64{1, 1}, nn.ASigmoid))
}
