package nn

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
)

// 神经网络包，原3NN.go内容

// 激活函数：Sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid 的导数
func sigmoidDerivative(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}

// 神经网络结构
type NeuralNetwork struct {
	inputNodes  int
	hiddenNodes int
	outputNodes int

	wInputHidden  [][]float64
	wHiddenOutput [][]float64

	bHidden []float64
	bOutput []float64

	learningRate float64
}

// 初始化神经网络
func NewNeuralNetwork(input, hidden, output int, lr float64) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())
	nn := &NeuralNetwork{
		inputNodes:    input,  //输入层
		hiddenNodes:   hidden, //隐藏层
		outputNodes:   output, //输出层
		learningRate:  lr,
		wInputHidden:  make([][]float64, input),  //输入层到隐藏层的权重
		wHiddenOutput: make([][]float64, hidden), //隐藏层到输出层的权重
		bHidden:       make([]float64, hidden),   //隐藏层偏置
		bOutput:       make([]float64, output),   //输出层偏置
	}
	// 可以部分并行化初始化权重和偏置
	var wg sync.WaitGroup

	for i := 0; i < input; i++ {
		nn.wInputHidden[i] = make([]float64, hidden)
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < hidden; j++ {
				nn.wInputHidden[i][j] = rand.NormFloat64()
			}
		}(i)
	}

	for i := 0; i < hidden; i++ {
		nn.wHiddenOutput[i] = make([]float64, output)
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < output; j++ {
				nn.wHiddenOutput[i][j] = rand.NormFloat64()
			}
			nn.bHidden[i] = rand.NormFloat64()
		}(i)
	}

	for i := 0; i < output; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			nn.bOutput[i] = rand.NormFloat64()
		}(i)
	}

	wg.Wait()
	return nn
}

// 前向传播
func (nn *NeuralNetwork) Predict(input []float64) []float64 {
	hidden := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		sum := nn.bHidden[i]
		for j := 0; j < nn.inputNodes; j++ {
			sum += input[j] * nn.wInputHidden[j][i]
		}
		hidden[i] = sigmoid(sum)
	}
	output := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		sum := nn.bOutput[i]
		for j := 0; j < nn.hiddenNodes; j++ {
			sum += hidden[j] * nn.wHiddenOutput[j][i]
		}
		output[i] = sigmoid(sum)
	}
	return output
}

// 通用前向传播函数
func ForwardLayer(input []float64, weights [][]float64, bias []float64, activation func(float64) float64) (z, a []float64) {
	nOut := len(bias)
	z = make([]float64, nOut)
	a = make([]float64, nOut)
	for i := 0; i < nOut; i++ {
		sum := bias[i]
		for j := 0; j < len(input); j++ {
			sum += input[j] * weights[j][i]
		}
		z[i] = sum
		a[i] = activation(sum)
	}
	return
}

// 通用反向传播函数
func BackwardLayer(
	errorNext []float64, // 下一层误差
	z []float64, // 本层加权和
	aPrev []float64, // 上一层输出
	weights [][]float64, // 本层权重
	activationDeriv func(float64) float64, // 激活函数导数
) (errorPrev []float64, gradW [][]float64, gradB []float64) {
	nIn := len(aPrev)
	nOut := len(z)
	gradW = make([][]float64, nIn)
	for i := 0; i < nIn; i++ {
		gradW[i] = make([]float64, nOut)
	}
	gradB = make([]float64, nOut)
	delta := make([]float64, nOut)
	for i := 0; i < nOut; i++ {
		delta[i] = errorNext[i] * activationDeriv(z[i])
		gradB[i] = delta[i]
		for j := 0; j < nIn; j++ {
			gradW[j][i] = aPrev[j] * delta[i]
		}
	}
	errorPrev = make([]float64, nIn)
	for i := 0; i < nIn; i++ {
		sum := 0.0
		for j := 0; j < nOut; j++ {
			sum += delta[j] * weights[i][j]
		}
		errorPrev[i] = sum
	}
	return
}

// 训练（单步）
func (nn *NeuralNetwork) Train(input, target []float64) {
	// 前向传播
	hiddenZ, hiddenA := ForwardLayer(input, nn.wInputHidden, nn.bHidden, sigmoid)//*Z表示加权和的结果，*A表示激活函数的结果
	outputZ, outputA := ForwardLayer(hiddenA, nn.wHiddenOutput, nn.bOutput, sigmoid)

	// 输出层误差
	outputError := make([]float64, nn.outputNodes)
	for i := 0; i < nn.outputNodes; i++ {
		outputError[i] = target[i] - outputA[i]
	}

	// 反向传播：输出层
	_, gradWHiddenOutput, gradBOutput := BackwardLayer(
		outputError, outputZ, hiddenA, nn.wHiddenOutput, sigmoidDerivative,
	)

	// 先计算隐藏层误差
	hiddenError := make([]float64, nn.hiddenNodes)
	for i := 0; i < nn.hiddenNodes; i++ {
		sum := 0.0
		for j := 0; j < nn.outputNodes; j++ {
			sum += outputError[j] * nn.wHiddenOutput[i][j] * sigmoidDerivative(hiddenZ[i])
		}
		hiddenError[i] = sum
	}
	// 用hiddenError调用BackwardLayer
	_, gradWInputHidden, gradBHidden := BackwardLayer(
		hiddenError, hiddenZ, input, nn.wInputHidden, sigmoidDerivative,
	)

	// 更新权重和偏置
	for i := 0; i < nn.hiddenNodes; i++ {
		for j := 0; j < nn.outputNodes; j++ {
			nn.wHiddenOutput[i][j] += nn.learningRate * gradWHiddenOutput[i][j]
		}
	}
	for i := 0; i < nn.outputNodes; i++ {
		nn.bOutput[i] += nn.learningRate * gradBOutput[i]
	}
	for i := 0; i < nn.inputNodes; i++ {
		for j := 0; j < nn.hiddenNodes; j++ {
			nn.wInputHidden[i][j] += nn.learningRate * gradWInputHidden[i][j]
		}
	}
	for i := 0; i < nn.hiddenNodes; i++ {
		nn.bHidden[i] += nn.learningRate * gradBHidden[i]
	}
}

// 可选：暴露一个简单的测试函数
func ExampleXOR() {
	nn := NewNeuralNetwork(2, 3, 1, 0.5)
	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	targets := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}
	for epoch := 0; epoch < 10000; epoch++ {
		for i := 0; i < len(data); i++ {
			nn.Train(data[i], targets[i])
		}
	}
	for i := 0; i < len(data); i++ {
		out := nn.Predict(data[i])
		fmt.Printf("输入: %v, 输出: %.4f\n", data[i], out[0])
	}
}
