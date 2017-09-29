#ifndef _neuralnetwork_hpp
#define _neuralnetwork_hpp

#include <cmath>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <cstdint>

using std::string;

//Holds the weights, is basically the neural network
struct neuralnet{
	//from -> to
	uint32_t ninputs, nhlayers, nhiddens, noutputs;
	float **weights_ih;//[INPUTS][HIDDENS];
	float ***weights_hh;//[HLAYERS-1][HIDDENS][HIDDENS];
	float **weights_ho;//[HIDDENS][OUTPUTS];
};

//holds the values of a node
struct Node{
	float netin;
	float activation;
};

//stores the values of the network for when propagating, in addition to holding the feeded values and outputvalues
struct calculationnet{
	Node *input;//[INPUTS];
	Node **hiddens;//[HLAYERS][HIDDENS];
	Node *output;//[OUTPUTS];
};

//Stores the partial derivative of the error of the nodes of a network in regard to its net input
struct dEdnetnet{
	float *input;//[INPUTS];
	float **hiddens;//[HLAYERS][HIDDENS];
	float *output;//[OUTPUTS];
};

//Stores the previous change in each weight, used for when backpropagating with momentum
struct dwnet{
	float **dweights_ih;//[INPUTS][HIDDENS];
	float ***dweights_hh;//[HLAYERS-1][HIDDENS][HIDDENS];
	float **dweights_ho;//[HIDDENS][OUTPUTS];
};

float actfunc(float n);

void feed(neuralnet &net, calculationnet &cnet, float *input);

void propagate(neuralnet &net, calculationnet &cnet);

void deriveerror(neuralnet &net, calculationnet &cnet, dEdnetnet& denet, float *exp_out);

void backpropagate(neuralnet &net, calculationnet &cnet, dEdnetnet &denet, dwnet &wnet, float learnrate, float momentum);

void randomize(neuralnet &net, float range);

void init(neuralnet &net, int _inputs, int _hlayers, int _hiddens, int _outputs);

void free(neuralnet &net);

void init(calculationnet &cnet, neuralnet &nnet);

void free(calculationnet &cnet, neuralnet &nnet);

void init(dEdnetnet &dnet, neuralnet &nnet);

void free(dEdnetnet &dnet, neuralnet &nnet);

void init(dwnet &dnet, neuralnet &nnet);

void free(dwnet &dnet, neuralnet &nnet);

bool save(neuralnet &net, string filename);

bool load(neuralnet &net, string filename);

#endif

