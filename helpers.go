package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/mat"
)

func Filter(ss []string, test func(string) bool) (ret []string) {
	for _, s := range ss {
		if test(s) {
			ret = append(ret, s)
		}
	}
	return
}

func Elapsed(what string) func() {
	start := time.Now()
	return func() {
		fmt.Printf("%s took %v\n", what, time.Since(start))
	}
}

func ToSlice(c chan *mat.Dense) []*mat.Dense {
	s := make([]*mat.Dense, 0)
	for i := range c {
		s = append(s, i)
	}
	return s
}
