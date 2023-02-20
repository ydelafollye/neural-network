package nn

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

type Matrix struct {
	rows int
	cols int
	Data [][]float64
}

func NewMatrix(rows, cols int) Matrix {
	Data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		Data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			Data[i][j] = 0
		}
	}
	return Matrix{rows, cols, Data}
}

func MatrixFromArray(arr []float64) Matrix {
	m := NewMatrix(len(arr), 1)
	for i := 0; i < len(arr); i++ {
		m.Data[i][0] = arr[i]
	}
	return m
}

func (m Matrix) MatrixToArray() []float64 {
	var arr []float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			arr = append(arr, m.Data[i][j])
		}
	}
	return arr
}
func (m Matrix) Show() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			s := fmt.Sprintf("%f ", m.Data[i][j])
			fmt.Print(s)
		}
		fmt.Print("\n")
	}
	fmt.Print("\n")

}

func (m *Matrix) Add(x interface{}) {
	switch value := x.(type) {
	case Matrix:
		if m.rows == value.rows && m.cols == value.cols {
			for i := 0; i < m.rows; i++ {
				for j := 0; j < m.cols; j++ {
					m.Data[i][j] += value.Data[i][j]
				}
			}
		} else {
			fmt.Println("Columns and Rows of A must match Columns and Rows of B.")
		}
	case float64:
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				m.Data[i][j] += value
			}
		}
	case int:
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				m.Data[i][j] += float64(value)
			}
		}
	default:
		fmt.Println("use type float64 or Matrix")
		os.Exit(1)
	}
}

func MatrixSubtract(m1, m2 Matrix) Matrix {
	if m1.rows != m2.rows || m1.cols != m2.cols {
		fmt.Println("m1 and m2 must have the size")
		os.Exit(1)
	}
	result := NewMatrix(m1.rows, m1.cols)
	for i := 0; i < result.rows; i++ {
		for j := 0; j < result.cols; j++ {
			result.Data[i][j] = m1.Data[i][j] - m2.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) Randomize() {
	rand.Seed(time.Now().UnixNano())
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.Data[i][j] = 1 - (rand.Float64() * 2)
		}
	}
}

func MatrixTranspose(m Matrix) Matrix {
	result := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func MatrixMultiply(m Matrix, x interface{}) Matrix {
	var result Matrix
	switch value := x.(type) {
	case Matrix:
		// Matrix Product
		// if m.cols == value.rows {
		result = NewMatrix(m.rows, value.cols)
		for i := 0; i < result.rows; i++ {
			for j := 0; j < result.cols; j++ {
				sum := 0.0
				for k := 0; k < m.cols; k++ {
					sum += m.Data[i][k] * value.Data[k][j]
				}
				result.Data[i][j] = sum
			}
		}
		// } else {
		// 	fmt.Println("Matrices size don't match : m1.cols must be egal to m2.rows")
		// 	os.Exit(1)
		// }
	case float64:
		// Scalar product
		result = NewMatrix(m.rows, m.cols)
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				result.Data[i][j] = m.Data[i][j] * value
			}
		}
	case int:
		// Scalar product
		result = NewMatrix(m.rows, m.cols)
		for i := 0; i < m.rows; i++ {
			for j := 0; j < m.cols; j++ {
				result.Data[i][j] = m.Data[i][j] * float64(value)
			}
		}
	default:
		fmt.Println("use type float64 or Matrix")
		os.Exit(1)
	}
	return result
}

func (m *Matrix) Mapper(f func(v float64) float64) {
	var val float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val = m.Data[i][j]
			m.Data[i][j] = f(val)
		}
	}
}

func MatrixMapper(m Matrix, f func(v float64) float64) Matrix {
	result := NewMatrix(m.rows, m.cols)
	var val float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			val = m.Data[i][j]
			result.Data[i][j] = f(val)
		}
	}
	return result
}

func MatrixCopy(m Matrix) Matrix {
	mcopy := NewMatrix(m.rows, m.cols)
	for i := 0; i < mcopy.rows; i++ {
		for j := 0; j < mcopy.cols; j++ {
			mcopy.Data[i][j] = m.Data[i][j]
		}
	}
	return mcopy
}
