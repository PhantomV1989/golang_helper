package main

import (
	"math"
	"math/rand"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// WorkerVectorizer returns 2d dense matrix, gen as generator function
func WorkerVectorizer(strChan <-chan string, wg *sync.WaitGroup, outChan chan<- *mat.Dense) {
	defer wg.Done()
	for i := range strChan {
		vec := VectorizeStringWithRnn(i, letters, vecLetters)
		outChan <- vec
	}
	println("worker done")
}

// GetDenseFrom2DFloatArr returns 2d dense matrix, gen as generator function
func GetDenseFrom2DFloatArr(arr [][]float64) *mat.Dense {
	row := len(arr)
	col := len(arr[0])
	var arr2 []float64
	for i := range arr {
		arr2 = append(arr2, arr[i]...)
	}
	return mat.NewDense(row, col, arr2)
}

// GetDense2D returns 2d dense matrix, gen as generator function
func GetDense2D(x, y int, gen func() float64) *mat.Dense {
	data := make([]float64, x*y)
	for i := range data {
		data[i] = gen()
	}
	return mat.NewDense(x, y, data)
}

// GetRandomDense2D returns 2d dense matrix
func GetRandomDense2D(x, y int) *mat.Dense {
	return GetDense2D(x, y, rand.Float64)
}

// VectorizeString returns an arr of *mat.Dense
func VectorizeString(toBeVectorized string, letters string, vecLetters *mat.Dense) *mat.Dense {
	var arr []float64
	for le := range toBeVectorized {
		pos := strings.Index(letters, toBeVectorized[le:le+1])
		arr = append(arr, mat.Row(nil, pos, vecLetters)...)
	}
	_, y := vecLetters.Dims()
	return mat.NewDense(len(toBeVectorized), y, arr)
}

// VectorizeStringWithRnn vectorize string to a fixed shape output
func VectorizeStringWithRnn(toBeVectorized string, letters string, vecLetters *mat.Dense) *mat.Dense {
	testVector := VectorizeString(toBeVectorized, letters, vecLetters)
	seed := GetDense2D(1, strEmbSize, func() float64 { return 0.0 })
	result := UseRNN(testVector, modelStrReader, seed)
	return result
}

// UseRNN returns an arr of *mat.Dense
func UseRNN(varMat *mat.Dense, rnn *mat.Dense, seed *mat.Dense) *mat.Dense {
	x, _ := varMat.Dims()
	for i := 0; i < x; i++ {
		_in := append(mat.Row(nil, i, varMat), mat.Row(nil, 0, seed)...)
		inMat := mat.NewDense(1, strReaderInputSize, _in)
		seed.Mul(inMat, rnn)
		DenseMapFunction(seed, FastSigmoid)
	}
	return seed
}

// FastSigmoid x / (1 + math.Abs(x))
func FastSigmoid(x float64) float64 {
	return x / (1 + math.Abs(x))
}

// DenseMapFunction value mapper, eg DenseMapFunction(seed, FastSigmoid)
func DenseMapFunction(inMat *mat.Dense, boo func(float64) float64) {
	foo := func(i, j int, v float64) float64 {
		return boo(inMat.At(i, j))
	}
	inMat.Apply(foo, inMat)
}

func StackDense(darr []*mat.Dense) *mat.Dense {
	_, c := darr[0].Dims()
	base := mat.NewDense(len(darr), c, nil)
	for i := range darr {
		base.SetRow(i, darr[i].RawRowView(0))
	}
	return base
}
