<h3>Multilayer Perceptron Neural Network Time-Series Predictor with the Back-Propagation Algorithm</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\TimeSeriesPredMLP\src\backprop\ directory and issue "go run ann.go" to
start the Multilayer Perceptron (MLP) Neural Network Time-Series Predictor server. In a web browser enter http://127.0.0.1:8080/mlptspredictor
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, samples consisting of noisy sine wave samples and the desired next noiseless sample are supplied to the network.
The network itself is a directed graph consisting of an input layer of nodes, one or more hidden layers of nodes, and
an output layer of nodes.  Each layer of nodes can be arbitrarily deep.  The nodes of the network are connected by weighted
links.  The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron in the hidden layer consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  The output layer is linear.  This means
there is no activation function.  The output is a weighted linear combination of the last hidden layer.  <b>The purpose of this program is to predict
the next output based on the previous inputs</b>.  The input consists of a sine wave and normally distributed noise at specified signal-to-noise ratio.
The user selects the MLP training parameters:
<li>Hidden Layers</li>
<li>Layer Depth</li>
<li>Training Samples</li>
<li>Sample Rate</li>
<li>Learning Rate</li>
<li>Momentum</li>
<li>Epochs</li>
<li>Prediction Order</li>
<li>Frequency</li>
<li>SNR</li>
<br>
<p>
The <i>Learning Rate</i> and <i>Momentum</i> must be less than one.  Each <i>Epoch</i> consists of the number of <i>Training Samples</i>.  
One training sample is Prediction-Order noisy sine wave samples  and the desired next noiseless sine wave sample.  The <i>Prediction Order</i>
is the number of previous samples of the input to use to predict the next sample.  The <i>Sample Rate</i> is the number of samples per second
or the sampling rate in Hz.  The <i>Frequency</i> is the frequency of the sine wave in Hz.  It must be less than half the Sample Rate, otherwise
aliasing will result.  The <i>SNR</i> is the signal-to-noise ratio in dB, which is 10*log10(signal power/noise power).  The sine wave is unit
amplitude which means the signal power is 0.5.  The noise power is the variance of the Normal random variable, which is determined by the SNR.
</p>
<p>
When the <i>Submit</i> button on the MLP Training Parameters form is clicked, the weights in the network are trained
and the mean-square error (MSE) is graphed versus Epoch.  This is the so-called <b>Learning Curve</b>.  As can be seen in the screen shots below, 
there is significant variance over the epochs.
</p>
<p>
When the <i>Test</i> link is clicked, 8,000 samples are generated, and vectors consisting of Prediction-Order noisy sine wave samples are supplied
to the MLP Neural Network.  The Neural Network predicts what the next sample should be.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the MLP.  In general, the more neurons in the
network, the more training examples will be needed to reduce the MSE to zero.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>

<h3>MLP Training, Learning Curve, Prediction Order = 50, SNR = 0dB, 5Hz, 3 Hidden Layers, Layer Depth = 20</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/eec6a60b-5a29-499a-a691-73d9e8820601)

 <h3>MLP Testing, Input Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/0d48b265-d958-467b-8495-1c0e12b7ec7b)

<h3>MLP Testing, Output Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/2a917bc3-3f8b-4ddc-9aeb-68d523106cdf)

<h3>MLP Training, Learning Curve, Prediction Order = 100, SNR=-10, 10Hz, 3 Hidden Layers, Layer Depth = 20</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/d51d2bbc-ccb1-4d7d-9098-ebaf28238b8e)

<h3>MLP Testing, Input Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/9b87a336-bc57-4218-9f45-d9be188ce764)

<h3>MLP Testing, Output Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/f61f810b-6c22-415f-991c-3552296c7fd1)

<h3>MLP Training, Learning Curve, Prediction Order = 50, SNR=10dB, 1 Hidden Layer, Layer Depth=20</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/0691f52b-cb19-4ac5-bd11-7d3a70c89aea)

<h3>MLP Testing, Input Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/6eeb29c4-649e-4bc9-aba2-eeed538754d4)

<h3>MLP Testing, Output Signal versus Time</h3>

![image](https://github.com/thomasteplick/mlptspredictor/assets/117768679/33df78a4-3db2-41d2-ba6c-ffac063538c7)




 
