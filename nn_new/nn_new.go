package nn_new





import (
	"math"
	"math/rand"
	"sync"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	s := sigmoid(x)
	return s * (1 - s)
}


//记录若干次的输入
type LayerInputPool struct{
	RecordNum int
	InputNum  int
	Inputs    [][]float64
	RecordIndex int//用来承接上一层的记录的索引，每一轮+1
	//那么当前这一轮前向传播使用的Index应该是RecordIndex-1
	//当前这一轮反向传播使用的Index应该是RecordIndex+1（在反向传播运用完之后Input就被丢弃了）
}

//其中layerIndex是从0开始计算，layersNum是总层数
func NewLayerInputPool(layerIndex int, layersNum int, InputNum int) *LayerInputPool {
	
	//计算recordNum
	recordNum := 2*(layersNum-layerIndex) + 1
	//根据recordNum和InputNum初始化二维切片
	Inputs := make([][]float64, recordNum)
	for i := 0; i < recordNum; i++ {
		Inputs[i] = make([]float64, InputNum)
	}

	return &LayerInputPool{
		RecordNum: recordNum,
		InputNum:  InputNum,
		Inputs:    Inputs,
	}
}


//单层神经网络的结构
type Layer struct {
	InputNum  int
	OutputNum int
	LayerIndex  int
	Weights   [][]float64
	Bias      []float64
	InputPool *LayerInputPool
}

// NewLayer 创建一个新的Layer，并初始化Weights和Bias
func NewLayer(InputNum int, OutputNum int, layerIndex int,layersNum int) *Layer {
	weights := make([][]float64, OutputNum)
	for i := 0; i < OutputNum; i++ {
		weights[i] = make([]float64, InputNum)
		for j := 0; j < InputNum; j++ {
			weights[i][j] = rand.NormFloat64() // 可根据需要选择初始化方式
		}
	}
	bias := make([]float64, OutputNum)
	for i := 0; i < OutputNum; i++ {
		bias[i] = rand.NormFloat64()
	}
	return &Layer{
		InputNum:  InputNum,
		OutputNum: OutputNum,
		Weights:   weights,
		Bias:      bias,
		LayerIndex: layerIndex,
		InputPool: NewLayerInputPool(layerIndex, layersNum, InputNum),
	}
}

// NewLayerParallel 并行初始化权重和偏置
func NewLayerParallel(InputNum, OutputNum int, layerIndex int, layersNum int) *Layer {
	weights := make([][]float64, OutputNum)
	bias := make([]float64, OutputNum)
	var wg sync.WaitGroup
	for i := 0; i < OutputNum; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			weights[i] = make([]float64, InputNum)
			for j := 0; j < InputNum; j++ {
				weights[i][j] = rand.NormFloat64()
			}
			bias[i] = rand.NormFloat64()
		}(i)
	}
	wg.Wait()
	return &Layer{
		InputNum:  InputNum,
		OutputNum: OutputNum,
		Weights:   weights,
		Bias:      bias,
		InputPool: NewLayerInputPool(layerIndex, layersNum, InputNum),
	}

}


// 前向传播
func (l *Layer) Forward(input []float64) ([]float64, []float64) {
	if len(input) != l.InputNum {
		panic("输入数据维度不匹配")
	}
	z := make([]float64, l.OutputNum)
	a := make([]float64, l.OutputNum)
	for i := 0; i < l.OutputNum; i++ {
		sum := l.Bias[i]
		for j := 0; j < l.InputNum; j++ {
			sum += input[j] * l.Weights[i][j]
		}
		z[i] = sum
		a[i] = sigmoid(sum)
	}
	return z, a
}

//后向传播
func (l *Layer) Backward(input, delta []float64, learningRate float64) []float64 {
	if len(input) != l.InputNum || len(delta) != l.OutputNum {
		panic("输入数据维度不匹配")
	}
	//计算上一层的delta
	prevDelta := make([]float64, l.InputNum)
	for i := 0; i < l.InputNum; i++ {
		sum := 0.0
		for j := 0; j < l.OutputNum; j++ {
			sum += delta[j] * l.Weights[j][i]
		}
		prevDelta[i] = sum * sigmoidDerivative(input[i])
	}
	//更新权重和偏置
	for i := 0; i < l.OutputNum; i++ {
		for j := 0; j < l.InputNum; j++ {
			l.Weights[i][j] -= learningRate * delta[i] * input[j]
		}
		l.Bias[i] -= learningRate * delta[i]
	}
	return prevDelta
}


