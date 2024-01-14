package main

import (
	"flag"
	"fmt"

	"github.com/mreza101/gonn/ch1/nn"
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

// Main function
func main() {
	// Create a new perceptron
	p := nn.NewPerceptron(2, nn.Step)
	// Train the perceptron
	p.TrainAll(trainingData, *epochs)
	// Test the perceptron
	fmt.Printf("0,0 = %f\n", p.FeedForward([]float64{0, 0}))
	fmt.Printf("0,1 = %f\n", p.FeedForward([]float64{0, 1}))
	fmt.Printf("1,0 = %f\n", p.FeedForward([]float64{1, 0}))
	fmt.Printf("1,1 = %f\n", p.FeedForward([]float64{1, 1}))
}
