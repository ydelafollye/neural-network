package nn

type NeuralNetwork struct {
	inputNodes  int
	hiddenNodes int
	outputNodes int

	WeightsInputHidden  Matrix
	weightsHiddenOutput Matrix
	biasHidden          Matrix
	biasOutput          Matrix

	LearningRate float64
}

func NewNeuralNetwork(inputNodes, hiddenNodes, outputNodes int) NeuralNetwork {
	wih := NewMatrix(hiddenNodes, inputNodes)
	who := NewMatrix(outputNodes, hiddenNodes)
	bh := NewMatrix(hiddenNodes, 1)
	bo := NewMatrix(outputNodes, 1)

	wih.Randomize()
	who.Randomize()
	bh.Randomize()
	bo.Randomize()

	lr := 0.1

	return NeuralNetwork{inputNodes, hiddenNodes, outputNodes, wih, who, bh, bo, lr}
}

func (n NeuralNetwork) Predict(inputArray []float64) []float64 {
	// Generating the Hidden Outputs
	inputs := MatrixFromArray(inputArray)
	hidden := MatrixMultiply(n.WeightsInputHidden, inputs)
	hidden.Add(n.biasHidden)

	// Activation function
	hidden.Mapper(sigmoid)

	// Generating the output
	output := MatrixMultiply(n.weightsHiddenOutput, hidden)
	output.Add(n.biasOutput)
	output.Mapper(sigmoid)

	return output.MatrixToArray()
}

func (n *NeuralNetwork) Train(inputArray []float64, targetArray []float64) {
	inputsMatrix := MatrixFromArray(inputArray)

	// Generating the Hidden Outputs
	hiddenMatrix := MatrixMultiply(n.WeightsInputHidden, inputsMatrix)
	hiddenMatrix.Add(n.biasHidden)

	// Activation function
	hiddenMatrix.Mapper(sigmoid)

	// Generating the output
	outputsMatrix := MatrixMultiply(n.weightsHiddenOutput, hiddenMatrix)
	outputsMatrix.Add(n.biasOutput)
	outputsMatrix.Mapper(sigmoid)

	targetsMatrix := MatrixFromArray(targetArray)
	// Calculate the error
	// error = targets - outputs
	outputErrors := MatrixSubtract(targetsMatrix, outputsMatrix)

	// Calculate gradient
	gradients := MatrixMapper(outputsMatrix, dsigmoid)
	gradients = MatrixMultiply(gradients, outputErrors)
	gradients = MatrixMultiply(gradients, n.LearningRate)

	// Calculate deltas
	hiddenMatrixTranspose := MatrixTranspose(hiddenMatrix)
	weightHiddenOutputDeltas := MatrixMultiply(gradients, hiddenMatrixTranspose)

	// Adjust the weights by deltas
	n.weightsHiddenOutput.Add(weightHiddenOutputDeltas)

	// Adjust the bias by its deltas (gradients)
	n.biasOutput.Add(gradients)

	// Calculate the hidden layer errors
	weightsHiddenOutputTranspose := MatrixTranspose(n.weightsHiddenOutput)
	hiddenErrors := MatrixMultiply(weightsHiddenOutputTranspose, outputErrors)

	// Calculate hidden gradient
	hiddenGradient := MatrixMapper(hiddenMatrix, dsigmoid)
	hiddenGradient = MatrixMultiply(hiddenGradient, hiddenErrors)
	hiddenGradient = MatrixMultiply(hiddenGradient, n.LearningRate)

	// Calculate input -> hidden deltas
	inputsMatrixTranspose := MatrixTranspose(inputsMatrix)
	weightInputHiddenDeltas := MatrixMultiply(hiddenGradient, inputsMatrixTranspose)
	n.WeightsInputHidden.Add(weightInputHiddenDeltas)

	// Adjust the bias by its deltas (gradients)
	n.biasHidden.Add(hiddenGradient)
}

// NeuroEvolution functions
func NeuralNetworkCopy(n NeuralNetwork) NeuralNetwork {
	var ncopy NeuralNetwork
	ncopy.inputNodes = n.inputNodes
	ncopy.hiddenNodes = n.hiddenNodes
	ncopy.outputNodes = n.outputNodes
	ncopy.LearningRate = n.LearningRate
	ncopy.WeightsInputHidden = MatrixCopy(n.WeightsInputHidden)
	ncopy.weightsHiddenOutput = MatrixCopy(n.weightsHiddenOutput)
	ncopy.biasHidden = MatrixCopy(n.biasHidden)
	ncopy.biasOutput = MatrixCopy(n.biasOutput)

	return NeuralNetwork{ncopy.inputNodes, ncopy.hiddenNodes, ncopy.outputNodes, ncopy.WeightsInputHidden, ncopy.weightsHiddenOutput, ncopy.biasHidden, ncopy.biasOutput, ncopy.LearningRate}
}

func (n *NeuralNetwork) Mutate() {
	n.weightsHiddenOutput.Mapper(mutate)
	n.WeightsInputHidden.Mapper(mutate)
	n.biasOutput.Mapper(mutate)
	n.biasHidden.Mapper(mutate)
}
