package main

import (
	"flag"
	"fmt"

	"github.com/mreza101/gonn/ch3/models"
	"github.com/mreza101/gonn/ch3/nn"
)

type data struct {
	Input   []float64
	Targets []float64
}

// trainingData for a logical OR
var trainingData = []data{
	{Input: []float64{0, 0}, Targets: []float64{0}},
	{Input: []float64{0, 1}, Targets: []float64{1}},
	{Input: []float64{1, 0}, Targets: []float64{1}},
	{Input: []float64{1, 1}, Targets: []float64{1}},
}

func getInputs(all []data) [][]float64 {
	var inputs [][]float64
	for _, d := range all {
		inputs = append(inputs, d.Input)
	}
	return inputs
}

func getTargets(all []data) [][]float64 {
	var targets [][]float64
	for _, d := range all {
		targets = append(targets, d.Targets)
	}
	return targets
}

var epochs = flag.Int("epochs", 10, "Number of epochs to train for")
var learningRate = flag.Float64("learning-rate", 0.1, "Learning rate to use")

func main() {
	// Create a new perceptron.
	model := models.NewNetwork()
	model.AddLayer(2, 1, nn.ASigmoid)

	// Train the perceptron.
	model.TrainAll(getInputs(trainingData), getTargets(trainingData), *epochs, *learningRate)

	// Test the perceptron.
	fmt.Printf("0,0 = %f\n", model.FeedForward([]float64{0, 0}))
	fmt.Printf("0,1 = %f\n", model.FeedForward([]float64{0, 1}))
	fmt.Printf("1,0 = %f\n", model.FeedForward([]float64{1, 0}))
	fmt.Printf("1,1 = %f\n", model.FeedForward([]float64{1, 1}))
}
