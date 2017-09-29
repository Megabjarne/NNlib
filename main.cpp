#include "neuralnetwork.hpp"
#include <iostream>
#include <time.h>

using namespace std;

int main(){
	srand(time(NULL));
	
	//we create all the necessary structs
	neuralnet net;
	calculationnet cnet;
	dwnet dnet;
	dEdnetnet dednet;
	
	//We create a network with 16 inputs, 2 hidden layers, 16 hidden nodes in each layers, 4 outputs
	init(net, 16, 2, 16, 4);
	//we randomize the weights to values between -1 and 1
	randomize(net, 1);
	
	//we initialize the calculationnet, dEdnet-net, and the dwnet
	init(cnet, net);
	init(dnet, net);
	init(dednet, net);
	
	//we create input and outputvectors, with the appropriate sizes
	float in[16];
	float out[4];
	
	while (true){
		//we pick a random value to train with
		int val = rand()%16;
		//we write a one to the val'th in-value and zeros to the others
		for (int k=0; k<16; k++){
			in[k] = (k == val);
		}
		//we write the binary pattern of the val to the outputvector
		for (int k=0; k<4;k++){
			out[k] = ((val & (1<<(3-k))) != 0);
		}
		
		//we "place" the inputvector in the calculation net
		feed(net, cnet, in);
		//we propagate the input through the network
		propagate(net, cnet);
		//we derive the error for each node in the network, has to be done in order to do backpropagation
		deriveerror(net, cnet, dednet, out);
		//we use what we found from the deriveerror-call and do the actual weight changing, with a certain learnrate and momentum
		backpropagate(net, cnet, dednet, dnet, 0.1, 0.8);
		
		//we write the input value, and the given output from the network
		cout<<val<<" -> "
			<<((int)(cnet.output[0].activation>0.5))
			<<((int)(cnet.output[1].activation>0.5))
			<<((int)(cnet.output[2].activation>0.5))
			<<((int)(cnet.output[3].activation>0.5))<<endl;
	}
	
	return 0;
}

