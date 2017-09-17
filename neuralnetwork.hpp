#ifndef _neuralnetwork_hpp
#define _neuralnetwork_hpp

#include <cmath>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <cerrno>

using std::string;
using std::cout;
using std::endl;

/*
#define INPUTS 286
#define HLAYERS 3
#define HIDDENS 286
#define OUTPUTS 256
#define LEARNRATE 0.05
#define MOMENTUM 0.95
*/

//nodes: 286 + 286*3 + 256 = 1400
//links: 286*286 + 2*286*286 + 286*256 = 245644

struct neuralnet{
	//from -> to
	int ninputs, nhlayers, nhiddens, noutputs;
	float **weights_ih;//[INPUTS][HIDDENS];
	float ***weights_hh;//[HLAYERS-1][HIDDENS][HIDDENS];
	float **weights_ho;//[HIDDENS][OUTPUTS];
};

struct Node{
	float netin;
	float activation;
	float dEdnet;
};

struct calculationnet{
	int ninputs, nhlayers, nhiddens, noutputs;
	Node *input;//[INPUTS];
	Node **hiddens;//[HLAYERS][HIDDENS];
	Node *output;//[OUTPUTS];
	
	float **dweights_ih;//[INPUTS][HIDDENS];
	float ***dweights_hh;//[HLAYERS-1][HIDDENS][HIDDENS];
	float **dweights_ho;//[HIDDENS][OUTPUTS];
};

float actfunc(float n);

void calculate(neuralnet &net, calculationnet &cnet);

void backpropagate(neuralnet &net, calculationnet &cnet, float* input, float* exp_out, float learnrate, float momentum);

void randomize(neuralnet &net, float range);

void init(neuralnet &net, int _inputs, int _hlayers, int _hiddens, int _outputs);

void free(neuralnet &net);

void init(calculationnet &cnet, int _inputs, int _hlayers, int _hiddens, int _outputs);

void free(calculationnet &cnet);

bool save(neuralnet &net, string filename);

bool load(neuralnet &net, string filename);

#endif

