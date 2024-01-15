package nn

func NewVector(n int) []float64 {
	return make([]float64, n)
}

func Add(a, b []float64) []float64 {
	result := NewVector(len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func Sub(a, b []float64) []float64 {
	result := NewVector(len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func Mul(a, b []float64) []float64 {
	result := NewVector(len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

func Dot(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func Scale(a []float64, s float64) []float64 {
	result := NewVector(len(a))
	for i := range a {
		result[i] = a[i] * s
	}
	return result
}

func NewMatrix(m, n int) [][]float64 {
	matrix := make([][]float64, m)
	for i := range matrix {
		matrix[i] = NewVector(n)
	}
	return matrix
}

func AddMatrix(a, b [][]float64) [][]float64 {
	result := NewMatrix(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func SubMatrix(a, b [][]float64) [][]float64 {
	result := NewMatrix(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result
}

func MulMatrix(a, b [][]float64) [][]float64 {
	result := NewMatrix(len(a), len(b[0]))
	for i := range a {
		for j := range b[0] {
			for k := range b {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result
}

// MulMatrixVector multiplies a matrix by a vector.
func MulMatrixVector(a [][]float64, b []float64) []float64 {
	result := NewVector(len(a))
	for i := range a {
		for j := range b {
			result[i] += a[i][j] * b[j]
		}
	}
	return result
}

// MulVectorMatrix multiplies a vector by a matrix.
func MulVectorMatrix(a []float64, b [][]float64) []float64 {
	result := NewVector(len(b[0]))
	for i := range b[0] {
		for j := range a {
			result[i] += a[j] * b[j][i]
		}
	}
	return result
}
