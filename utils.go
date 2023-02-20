package nn

import (
	"math"
	"math/rand"
)

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func dsigmoid(y float64) float64 {
	return y * (1 - y)
}

func doubleIt(x float64) float64 {
	return x * 2
}

func mutate(val float64) float64 {
	if rand.Float64() < 0.05 {
		// return 1 - (rand.Float64() * 2)
		return val + rand.NormFloat64()*0.1 + 0
	}
	return val
}
