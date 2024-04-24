/*
Neural Network (nn) using multilayer perceptron architecture
and the backpropagation algorithm.  This is a web application that uses
the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/mlpbackprop.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the nn.  Each example consists
of an input vector of noisy time samples and a desired predicted output.  The nn
itself consists of an input layer of nodes, one or more hidden layers of nodes,
and an output layer of nodes.  The nodes are connected by weighted links.  The
weights are trained by back propagating the output layer errors forward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.

The hidden layers use a sigmoid activation function, and the output layer is linear.

The input vector is a time series consisting of a sinsuoid in white Gaussian noise.
The desired signal is a prediction of what the next sample should be.  The neural
network is a one-step predictor.  The user chooses the prediction order (the number
of past samples), SNR, and signal frequency in Hz.

*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTrainingMLP    = "templates/trainingMLP.html" // html for training MLP
	fileTestingMLP     = "templates/testingMLP.html"  // html for testing MLP
	patternTrainingMLP = "/mlptspredictor"            // http handler for training the MLP
	patternTestingMLP  = "/mlptspredictortest"        // http handler for testing the MLP
	rows               = 300                          // #rows in grid
	columns            = rows                         // #columns in grid
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	fileweights        = "weights.csv"                // mlp weights
	a                  = 1.7159                       // activation function const
	b                  = 2.0 / 3.0                    // activation function const
	K1                 = b / a
	K2                 = a * a
	maxClasses         = 25
	testingSamples     = 8000
	dataDir            = "data/" // directory for the weights
	twoPi              = 2.0 * math.Pi
	mlpoutput          = "mlpOutput.txt" // MLP predictor output from testing
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid            []string // plotting grid
	Status          string   // status of the plot
	Xlabel          []string // x-axis labels
	Ylabel          []string // y-axis labels
	Xmin            string   // x minimum endpoint in Euclidean graph
	Xmax            string   // x maximum endpoint in Euclidean graph
	Ymin            string   // y minimum endpoint in Euclidean graph
	Ymax            string   // y maximum endpoint in Euclidean graph
	HiddenLayers    string   // number of hidden layers
	LayerDepth      string   // number of Nodes in hidden layers
	LearningRate    string   // size of weight update for each iteration
	Momentum        string   // previous weight update scaling factor
	Epochs          string   // number of epochs
	TrainingSamples string   // number of training samples
	TestingSamples  string   // number of testing samples
	SampleRate      string   // sample rate in Hz
	PredictionOrder string   // number of past samples, input layer depth
	Frequency       string   // frequency of the sinusoid in Hz
	SNR             string   // signal-to-noise ratio in dB
}

// Type to hold the minimum and maximum data values of the Euclidean graph
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// graph links
type Link struct {
	wgt      float64 // weight
	wgtDelta float64 // previous weight update used in momentum
}

// Primary data structure for holding the MLP Backprop state
type MLP struct {
	plot            *PlotT   // data to be distributed in the HTML template
	Endpoints                // embedded struct
	link            [][]Link // links in the graph
	node            [][]Node // nodes in the graph
	samples         []float64
	mse             []float64 // mean square error in output layer per epoch
	epochs          int       // number of epochs
	learningRate    float64   // learning rate parameter
	momentum        float64   // delta weight scale constant
	hiddenLayers    int       // number of hidden layers
	desired         float64   // desired output of the sample
	layerDepth      int       // hidden layer number of nodes
	trainingSamples int       // number of training samples
	testingSamples  int       // number of testing samples
	predictionOrder int       // number of past samples in input vector
	frequency       int       // signal frequency in Hz
	snr             int       // signal-to-noise ratio in dB
	sampleRate      int       // sample rate in Hz or samples/sec
}

// global variables for parse and execution of the html template
var (
	tmplTrainingMLP *template.Template
	tmplTestingMLP  *template.Template
)

// calculateMSE calculates the MSE at the output layer every epoch termination
func (mlp *MLP) calculateMSE(epoch int) {
	// There is one output node.
	var err float64 = 0.0
	outputLayer := mlp.hiddenLayers + 1
	// Calculate (desired - mlp.node[L][n].y)^2 and store in mlp.mse[n]
	err = float64(mlp.desired) - mlp.node[outputLayer][0].y
	err2 := err * err
	mlp.mse[epoch] = err2

	// calculate min/max mse
	if mlp.mse[epoch] < mlp.ymin {
		mlp.ymin = mlp.mse[epoch]
	}
	if mlp.mse[epoch] > mlp.ymax {
		mlp.ymax = mlp.mse[epoch]
	}
}

// propagateForward the input vector starting with samp to output layer
func (mlp *MLP) propagateForward(samp int) error {
	// Assign sample to input layer, layer 0.  Remember first node is bias = 1
	i := 1
	d := len(mlp.node[0])
	for j := 1; j < d; j++ {
		mlp.node[0][i].y = mlp.samples[samp]
		i++
		samp++
	}

	// Assign the future (predicted) sample which is the desired output.
	// It is the next sample in the samples list.
	mlp.desired = mlp.samples[samp]

	// Loop over layers: mlp.hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer <= mlp.hiddenLayers; layer++ {
		// Loop over nodes in the layer, d1 is the layer depth of current
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Each node in previous layer is connected to current node because
			// the network is fully connected.  d2 is the layer depth of previous
			d2 := len(mlp.node[layer-1])
			// Loop over weights to get v
			v := 0.0
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				v += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer-1][i2].y
			}
			// compute output y = Phi(v)
			mlp.node[layer][i1].y = a * math.Tanh(b*v)
		}
	}

	// Last layer is different because there is no bias node, so the indexing is different.
	// Also, output is linear, no sigmoid activation function is used.
	layer := mlp.hiddenLayers + 1
	d1 := len(mlp.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		// Each node in previous layer is connected to current node because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(mlp.node[layer-1])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer-1][i2].y
		}
		// compute output y = v, linear output
		mlp.node[layer][i1].y = v
	}

	return nil
}

// propagateBackward the error from output to input layer
func (mlp *MLP) propagateBackward() error {

	// Output layer is different, no bias node, so the indexing is different.
	// Also, output layer is linear, no sigmoid activation function.
	// Loop over nodes in output layer
	layer := mlp.hiddenLayers + 1
	d1 := len(mlp.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		//compute error e=d-v
		mlp.node[layer][i1].delta = mlp.desired - mlp.node[mlp.hiddenLayers+1][i1].y
		// Linear output, y = v, and derivative of y = 1
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(mlp.node[layer-1])
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*d1+i1].wgt * mlp.node[layer][i1].delta
			// Compute weight delta, Update weight with momentum, y, and local gradient
			wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
			mlp.link[layer-1][i2*d1+i1].wgt +=
				wgtDelta + mlp.momentum*mlp.link[layer-1][i2*d1+i1].wgtDelta
			// update weight delta
			mlp.link[layer-1][i2*d1+i1].wgtDelta = wgtDelta

		}
		// Reset this local gradient to zero for next training example
		mlp.node[layer][i1].delta = 0.0
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := mlp.hiddenLayers; layer > 0; layer-- {
		// Loop over nodes in this layer, d1 is the current layer depth
		d1 := len(mlp.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Multiply deltas propagated from past node by this node's Phi'(v) to get local gradient.
			mlp.node[layer][i1].delta *= K1 * (K2 - mlp.node[layer][i1].y*mlp.node[layer][i1].y)
			// Send this node's local gradient to previous layer nodes through corresponding link.
			// Each node in previous layer is connected to current node because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(mlp.node[layer-1])
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				mlp.node[layer-1][i2].delta += mlp.link[layer-1][i2*(d1-1)+i1-1].wgt * mlp.node[layer][i1].delta
				// Compute weight delta, Update weight with momentum, y, and local gradient
				// anneal learning rate parameter: mlp.learnRate/(epoch*layer)
				// anneal momentum: momentum/(epoch*layer)
				wgtDelta := mlp.learningRate * mlp.node[layer][i1].delta * mlp.node[layer-1][i2].y
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgt +=
					wgtDelta + mlp.momentum*mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta
				// update weight delta
				mlp.link[layer-1][i2*(d1-1)+i1-1].wgtDelta = wgtDelta

			}
			// Reset this local gradient to zero for next training example
			mlp.node[layer][i1].delta = 0.0
		}
	}
	return nil
}

// runEpochs performs forward and backward propagation over each sample
func (mlp *MLP) runEpochs() error {
	for n := 0; n < mlp.epochs; n++ {

		// Create new training samples for every epoch; we cannot shuffle
		// the samples since they are a time series  and we are trying to
		// predict the next sample based on previous samples.
		err := mlp.createSamples(mlp.trainingSamples)
		if err != nil {
			fmt.Printf("createSamples error: %v\n", err)
			return fmt.Errorf("createSamples error: %v", err.Error())
		}

		// Loop over the training examples
		for k := range mlp.samples[:mlp.trainingSamples] {
			// Forward Propagation
			err := mlp.propagateForward(k)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}

			// Backward Propagation
			err = mlp.propagateBackward()
			if err != nil {
				return fmt.Errorf("backward propagation error: %s", err.Error())
			}
		}

		// At the end of each epoch, loop over the output nodes and calculate mse
		mlp.calculateMSE(n)

	}

	return nil
}

// init parses the html template files
func init() {
	tmplTrainingMLP = template.Must(template.ParseFiles(fileTrainingMLP))
	tmplTestingMLP = template.Must(template.ParseFiles(fileTestingMLP))
}

// createSamples creates a slice of training or testing samples
func (mlp *MLP) createSamples(samples int) error {
	// sinusoidal signal has amplitude 1, power = 0.5
	s := .5
	// noise standard deviation from the SNR
	noiseSigma := math.Sqrt(s / (math.Pow(10.0, float64(mlp.snr)/10.0)))
	for k := 0; k < samples+mlp.predictionOrder; k++ {
		// signal consists of sinusoid + Gaussian noise with standard deviation from the SNR
		mlp.samples[k] = math.Sin(twoPi*float64(mlp.frequency)/float64(mlp.sampleRate)*float64(k)) +
			noiseSigma*rand.NormFloat64()
	}
	return nil
}

// newMLP constructs an MLP instance
func newMLP(r *http.Request, plot *PlotT, hiddenLayers int) (*MLP, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("layerdepth")
	layerDepth, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("layerdepth int conversion error: %v\n", err)
		return nil, fmt.Errorf("layerdepth int conversion error: %s", err.Error())
	}

	txt = r.FormValue("trainingsamples")
	trainingSamples, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("trainingsamples int conversion error: %v\n", err)
		return nil, fmt.Errorf("trainingsamples int conversion error: %s", err.Error())
	}

	txt = r.FormValue("samplerate")
	sampleRate, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("samplerate int conversion error: %v\n", err)
		return nil, fmt.Errorf("samplerate int conversion error: %s", err.Error())
	}

	txt = r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	txt = r.FormValue("momentum")
	momentum, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("momentum float conversion error: %v\n", err)
		return nil, fmt.Errorf("momentum float conversion error: %s", err.Error())
	}

	txt = r.FormValue("epochs")
	epochs, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("epochs int conversion error: %v\n", err)
		return nil, fmt.Errorf("epochs int conversion error: %s", err.Error())
	}

	txt = r.FormValue("predictionorder")
	predictionOrder, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("predictionorder int conversion error: %v\n", err)
		return nil, fmt.Errorf("predictionorder int conversion error: %s", err.Error())
	}

	txt = r.FormValue("frequency")
	frequency, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("frequency int conversion error: %v\n", err)
		return nil, fmt.Errorf("frequency int conversion error: %s", err.Error())
	}

	txt = r.FormValue("snr")
	snr, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("snr int conversion error: %v\n", err)
		return nil, fmt.Errorf("snr int conversion error: %s", err.Error())
	}

	mlp := MLP{
		hiddenLayers:    hiddenLayers,
		layerDepth:      layerDepth,
		trainingSamples: trainingSamples,
		epochs:          epochs,
		learningRate:    learningRate,
		momentum:        momentum,
		plot:            plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples:         make([]float64, trainingSamples+predictionOrder),
		sampleRate:      sampleRate,
		predictionOrder: predictionOrder,
		frequency:       frequency,
		snr:             snr,
	}

	// construct link that holds the weights and weight deltas and initialize them
	mlp.link = make([][]Link, hiddenLayers+1)

	// input layer
	mlp.link[0] = make([]Link, 3*layerDepth)

	// one output layer node
	olnodes := 1
	// output layer links
	mlp.link[len(mlp.link)-1] = make([]Link, olnodes*(layerDepth+1))

	// hidden layer links
	for i := 1; i < len(mlp.link)-1; i++ {
		mlp.link[i] = make([]Link, (layerDepth+1)*layerDepth)
	}

	// Initialize the weights one time at the start of the training

	// input layer
	// initialize the wgt and wgtDelta randomly, zero mean, normalize by fan-in
	for i := range mlp.link[0] {
		mlp.link[0][i].wgt = 2.0 * (rand.ExpFloat64() - .5) / float64(predictionOrder)
		mlp.link[0][i].wgtDelta = 2.0 * (rand.ExpFloat64() - .5) / float64(predictionOrder)
	}

	// output layer links
	for i := range mlp.link[mlp.hiddenLayers] {
		mlp.link[mlp.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		mlp.link[mlp.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
	}

	// hidden layers
	for lay := 1; lay < len(mlp.link)-1; lay++ {
		for link := 0; link < len(mlp.link[lay]); link++ {
			mlp.link[lay][link].wgt = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
			mlp.link[lay][link].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(mlp.layerDepth)
		}
	}

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, 3)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// output layer, which has no bias node
	mlp.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= hiddenLayers; i++ {
		mlp.node[i] = make([]Node, layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	// mean-square error
	mlp.mse = make([]float64, epochs)

	return &mlp, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (mlp *MLP) gridFillInterp(op string) error {
	var (
		x            float64
		xstep        float64
		y            float64
		prevX, prevY float64
		xscale       float64
		yscale       float64
		data         []float64
		dlen         int
	)
	// Which container to use depends on whether training or testing.
	// training
	if op == "mse" {
		data = mlp.mse
		dlen = mlp.epochs
		xstep = 1.0
		// testing, input or output signal
	} else {
		data = mlp.samples
		dlen = mlp.testingSamples
		xstep = 1.0 / float64(mlp.sampleRate)
	}
	x = mlp.xmin
	y = mlp.ymin

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = (columns - 1) / (mlp.xmax - mlp.xmin)
	yscale = (rows - 1) / (mlp.ymax - mlp.ymin)

	mlp.plot.Grid = make([]string, rows*columns)

	// This cell location (row,col) is on the line
	row := int((mlp.ymax-y)*yscale + .5)
	col := int((x-mlp.xmin)*xscale + .5)
	mlp.plot.Grid[row*columns+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := mlp.ymax - mlp.ymin
	lenEPx := mlp.xmax - mlp.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < dlen; i++ {

		// next sample location
		x += xstep
		y = data[i]

		// This cell location (row,col) is on the line
		row := int((mlp.ymax-y)*yscale + .5)
		col := int((x-mlp.xmin)*xscale + .5)
		mlp.plot.Grid[row*columns+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(columns * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(rows * lenEdgeY / lenEPy)    // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((mlp.ymax-interpY)*yscale + .5)
			col := int((interpX-mlp.xmin)*xscale + .5)
			mlp.plot.Grid[row*columns+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (mlp *MLP) insertLabels() {
	mlp.plot.Xlabel = make([]string, xlabels)
	mlp.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (mlp.xmax - mlp.xmin) / (xlabels - 1)
	x := mlp.xmin
	// First label is empty for alignment purposes
	for i := range mlp.plot.Xlabel {
		mlp.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (mlp.ymax - mlp.ymin) / (ylabels - 1)
	y := mlp.ymin
	for i := range mlp.plot.Ylabel {
		mlp.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingMLP(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		mlp  *MLP
	)

	// Get the number of hidden layers
	txt := r.FormValue("hiddenlayers")
	// Need hidden layers to continue
	if len(txt) > 0 {
		hiddenLayers, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Hidden Layers int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Hidden Layers conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create MLP instance to hold state
		mlp, err = newMLP(r, &plot, hiddenLayers)
		if err != nil {
			fmt.Printf("newMLP() error: %v\n", err)
			plot.Status = fmt.Sprintf("newMLP() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the epochs
		err = mlp.runEpochs()
		if err != nil {
			fmt.Printf("runEpochs() error: %v\n", err)
			plot.Status = fmt.Sprintf("runEpochs() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put ensemble-averaged MSE vs Epoch in PlotT
		err = mlp.gridFillInterp("mse")
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		mlp.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
		mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
		mlp.plot.TrainingSamples = strconv.Itoa(mlp.trainingSamples)
		mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', 3, 64)
		mlp.plot.SampleRate = strconv.FormatFloat(float64(mlp.sampleRate), 'f', 0, 64)
		mlp.plot.PredictionOrder = strconv.Itoa(mlp.predictionOrder)
		mlp.plot.Frequency = strconv.Itoa(mlp.frequency)
		mlp.plot.SNR = strconv.Itoa(mlp.snr)
		mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', 3, 64)
		mlp.plot.Epochs = strconv.Itoa(mlp.epochs)

		// Save hiddenLayers, layerDepth, sampleRate, learningRate, momentum,
		// epochs, predictionOrder, snr, frequency
		// and weights to csv file, one layer per line
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(dataDir, fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(dataDir, fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingMLP.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()
		fmt.Fprintf(f, "%d,%d,%d,%f,%f,%d,%d,%d,%d\n",
			mlp.hiddenLayers, mlp.layerDepth, mlp.sampleRate, mlp.learningRate, mlp.momentum,
			mlp.epochs, mlp.predictionOrder, mlp.snr, mlp.frequency)
		for _, layer := range mlp.link {
			for _, node := range layer {
				fmt.Fprintf(f, "%f,", node.wgt)
			}
			fmt.Fprintln(f)
		}

		mlp.plot.Status = "MLP Neural Network created and MSE plotted"

		// Execute data on HTML template
		if err = tmplTrainingMLP.Execute(w, mlp.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Multilayer Perceptron (MLP) training parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// Predict test samples and display input/output of MLP
func (mlp *MLP) runPrediction(signal string) error {

	// Determine what is to be plotted
	if signal == "inputsignal" {
		// Loop over the training examples and find min/max
		for _, samp := range mlp.samples[:mlp.testingSamples] {
			if samp < mlp.ymin {
				mlp.ymin = samp
			}
			if samp > mlp.ymax {
				mlp.ymax = samp
			}
		}
		// plot the input signal of the MLP
		mlp.gridFillInterp("samples")
		mlp.plot.Status = "Input signal of MLP plotted."
	} else if signal == "outputsignal" {
		outputLayer := mlp.hiddenLayers + 1
		var y float64
		f, err := os.Create(path.Join(dataDir, mlpoutput))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(dataDir, mlpoutput), err)
			return fmt.Errorf("os.Create() file %s error: %v", path.Join(dataDir, mlpoutput), err.Error())
		}
		// Loop over the training examples and find min/max
		for samp := range mlp.samples[:mlp.testingSamples] {
			// Forward Propagation
			err := mlp.propagateForward(samp)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}
			// check for ymin and ymax
			y = mlp.node[outputLayer][0].y
			if y < mlp.ymin {
				mlp.ymin = y
			}
			if y > mlp.ymax {
				mlp.ymax = y
			}
			// save to disk
			fmt.Fprintf(f, "%f\n", y)
		}
		f.Close()
		fmt.Printf("ymin = %f, ymax = %f\n", mlp.ymin, mlp.ymax)

		// read in the MLP output and insert in the samples list
		f, err = os.Open(path.Join(dataDir, mlpoutput))
		if err != nil {
			fmt.Printf("Open file %s error: %v", mlpoutput, err)
			return fmt.Errorf("open file %s error: %s", mlpoutput, err.Error())
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		i := 0
		// Retrieve the MLP output and insert back into the samples list
		for scanner.Scan() {
			outstr := scanner.Text()
			out, err := strconv.ParseFloat(outstr, 64)
			if err != nil {
				fmt.Printf("Convert to float %s error: %v", outstr, err)
				return fmt.Errorf("convert to float %s error: %s", outstr, err.Error())
			}
			mlp.samples[i] = out
			i++
		}
		if err = scanner.Err(); err != nil {
			fmt.Printf("scanner error: %s", err.Error())
		}

		// plot the output signal of the MLP
		mlp.gridFillInterp("samples")

		mlp.plot.Status = "Prediction completed and output signal of MLP plotted."
	} else {
		fmt.Printf("Plotting signal (input or output) not chosen.")
		return fmt.Errorf("choose input or output signal to be plotted")
	}

	// At the end of all epochs, insert form previous control items in PlotT
	mlp.plot.HiddenLayers = strconv.Itoa(mlp.hiddenLayers)
	mlp.plot.LayerDepth = strconv.Itoa(mlp.layerDepth)
	mlp.plot.TrainingSamples = strconv.Itoa(mlp.trainingSamples)
	mlp.plot.LearningRate = strconv.FormatFloat(mlp.learningRate, 'f', 3, 64)
	mlp.plot.SampleRate = strconv.FormatFloat(float64(mlp.sampleRate), 'f', 0, 64)
	mlp.plot.PredictionOrder = strconv.Itoa(mlp.predictionOrder)
	mlp.plot.Frequency = strconv.Itoa(mlp.frequency)
	mlp.plot.SNR = strconv.Itoa(mlp.snr)
	mlp.plot.Momentum = strconv.FormatFloat(mlp.momentum, 'f', 3, 64)
	mlp.plot.Epochs = strconv.Itoa(mlp.epochs)
	mlp.plot.TestingSamples = strconv.Itoa(mlp.testingSamples)

	return nil
}

// newTestingMLP constructs an MLP from the saved mlp weights and parameters
func newTestingMLP(plot *PlotT) (*MLP, error) {
	// Read in weights from csv file, ordered by layers, and MLP parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	// hiddenLayers, layerDepth, sampleRate, learningRate, momentum,
	// epochs, predictionOrder, snr, frequency
	items := strings.Split(line, ",")
	hiddenLayers, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[0], err)
		return nil, err
	}
	layerDepth, err := strconv.Atoi(items[1])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[1], err)
		return nil, err
	}
	sampleRate, err := strconv.Atoi(items[2])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[2], err)
		return nil, err
	}
	learningRate, err := strconv.ParseFloat(items[3], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[3], err)
		return nil, err
	}
	momentum, err := strconv.ParseFloat(items[4], 64)
	if err != nil {
		fmt.Printf("Conversion to float of %s error: %v\n", items[4], err)
		return nil, err
	}
	epochs, err := strconv.Atoi(items[5])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[5], err)
		return nil, err
	}
	predictionOrder, err := strconv.Atoi(items[6])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[6], err)
		return nil, err
	}
	snr, err := strconv.Atoi(items[7])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[7], err)
	}
	frequency, err := strconv.Atoi(items[8])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v\n", items[8], err)
		return nil, err
	}

	// construct the testing mlp
	mlp := MLP{
		hiddenLayers:    hiddenLayers,
		layerDepth:      layerDepth,
		sampleRate:      sampleRate,
		learningRate:    learningRate,
		momentum:        momentum,
		epochs:          epochs,
		predictionOrder: predictionOrder,
		snr:             snr,
		frequency:       frequency,
		testingSamples:  testingSamples,
		plot:            plot,
		samples:         make([]float64, testingSamples+predictionOrder),
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(testingSamples) / float64(sampleRate)},
	}

	// retrieve the weights
	rows := 0
	for scanner.Scan() {
		rows++
		line = scanner.Text()
		weights := strings.Split(line, ",")
		weights = weights[:len(weights)-1]
		temp := make([]Link, len(weights))
		for i, wtStr := range weights {
			wt, err := strconv.ParseFloat(wtStr, 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", wtStr, err)
				continue
			}
			temp[i] = Link{wgt: wt, wgtDelta: 0}
		}
		mlp.link = append(mlp.link, temp)
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s", err.Error())
	}

	fmt.Printf("\nhidden layer depth = %d, hidden layers = %d\n"+
		"Testing Samples = %d, Sample Rate = %d, Learning Rate = %f\n"+
		"Momentum = %f, Epochs = %d, Prediction Order = %d, Frequency = %d, SNR = %d\n",
		mlp.layerDepth, mlp.hiddenLayers, mlp.testingSamples,
		mlp.sampleRate, mlp.learningRate, mlp.momentum, mlp.epochs,
		mlp.predictionOrder, mlp.frequency, mlp.snr)

	// construct node, init node[i][0].y to 1.0 (bias)
	mlp.node = make([][]Node, mlp.hiddenLayers+2)

	// input layer
	mlp.node[0] = make([]Node, 3)
	// set first node in the layer (bias) to 1
	mlp.node[0][0].y = 1.0

	// output layer, which has no bias node
	mlp.node[mlp.hiddenLayers+1] = make([]Node, 1)

	// hidden layers
	for i := 1; i <= mlp.hiddenLayers; i++ {
		mlp.node[i] = make([]Node, mlp.layerDepth+1)
		// set first node in the layer (bias) to 1
		mlp.node[i][0].y = 1.0
	}

	return &mlp, nil
}

// handleTesting performs pattern classification of the test data
func handleTestingMLP(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		mlp  *MLP
		err  error
	)
	// Construct MLP instance containing MLP state
	mlp, err = newTestingMLP(&plot)
	if err != nil {
		fmt.Printf("newTestingMLP() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingMLP() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Create testing samples
	err = mlp.createSamples(mlp.testingSamples)
	if err != nil {
		fmt.Printf("createSamples error: %v\n", err)
		plot.Status = fmt.Sprintf("createSamples error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Determine what needs to be plotted
	signal := r.FormValue("signal")
	// At end of all samples, plot input or output signals of MLP
	err = mlp.runPrediction(signal)
	if err != nil {
		fmt.Printf("runPrediction() error: %v\n", err)
		plot.Status = fmt.Sprintf("runPrediction() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTestingMLP.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// insert x-labels and y-labels in PlotT
	mlp.insertLabels()

	// Execute data on HTML template
	if err = tmplTestingMLP.Execute(w, mlp.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the MLP ANN

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingMLP, handleTrainingMLP)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingMLP, handleTestingMLP)
	fmt.Printf("Multilayer Perceptron Neural Network Time-Series Predictor Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
