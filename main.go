package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

var letters = " !\"#$%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~®\n"
var charEmbSize = 15
var strEmbSize = 100
var strReaderInputSize = charEmbSize + strEmbSize
var vecLetters = GetRandomDense2D(len(letters), charEmbSize)
var modelStrReader = GetDense2D(strReaderInputSize, strEmbSize, func() float64 { return rand.Float64()*2 - 1 })

func main() {
	testScore()
}

func test() {
	var arr [][]float64
	z1 := VectorizeStringWithRnn("rrrrrrrrrrrrrra", letters, vecLetters)
	z2 := VectorizeStringWithRnn("rrrrrrrrrrrrrrb", letters, vecLetters)
	z3 := VectorizeStringWithRnn("crrrrrrrrrrrrrr", letters, vecLetters)

	arr = append(arr, mat.Row(nil, 0, z1))
	arr = append(arr, mat.Row(nil, 0, z2))
	arr = append(arr, mat.Row(nil, 0, z3))

	var s string
	for i := range arr {
		for j := range arr[i] {
			s += fmt.Sprintf("%f", arr[i][j]) + " "
		}
		s += "\n"
	}

	d1 := []byte(s)
	err := ioutil.WriteFile("dat1", d1, 0644)
	_ = z1
	_ = z2
	_ = z3
	_ = err
}

func testNormalize() {
	var arr [][]float64
	z1 := VectorizeStringWithRnn("rrrrrrrrrrrrrra", letters, vecLetters)
	z2 := VectorizeStringWithRnn("rrrrrrrrrrrrrrb", letters, vecLetters)
	z3 := VectorizeStringWithRnn("crrrrrrrrrrrrrr", letters, vecLetters)

	arr = append(arr, mat.Row(nil, 0, z1))
	arr = append(arr, mat.Row(nil, 0, z2))
	arr = append(arr, mat.Row(nil, 0, z3))

	arrDense, ggg := Normalize(GetDenseFrom2DFloatArr(arr))
	_ = z1
	_ = z2
	_ = arrDense
	_ = ggg
}

func testScore() {
	bb, err := ioutil.ReadFile("data/sampleStrings")
	bbs := string(bb)
	bbarr := strings.Split(bbs, "\r\n")

	foo := func(s string) bool {
		if len(s) > 0 {
			return true
		}
		return false
	}

	bbarr = Filter(bbarr, foo)
	var wg sync.WaitGroup
	strChannel := make(chan string, 100000)
	outChan := make(chan *mat.Dense, 100000)
	defer Elapsed("testScore")()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go WorkerVectorizer(strChannel, &wg, outChan)
	}

	for j := range bbarr[:23] {
		strChannel <- bbarr[j]
	}
	close(strChannel)
	wg.Wait()
	close(outChan)
	println("done")

	arrDense := ToSlice(outChan)
	strDenseVec := StackDense(arrDense)
	dvectors, _ := Normalize(strDenseVec)
	// GetHierarchicalPC(dvectors)
	_ = err
	_ = bbs
	_ = bbarr
	_ = dvectors
}

func test1() bool {
	testVector := VectorizeString("asd", letters, vecLetters)
	qwe := mat.Row(nil, 0, testVector)
	asd := mat.Row(nil, 66, vecLetters)
	var br = true
	for i := range qwe {
		if qwe[i] != asd[i] {
			br = false
		}
	}
	return br
}
