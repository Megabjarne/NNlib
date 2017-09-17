#include "neuralnetwork.hpp"

/*
#define INPUTS 286
#define HLAYERS 3
#define HIDDENS 286
#define OUTPUTS 256
#define LEARNRATE 0.25

struct neuralnet{
	//from -> to
	float weights_ih[INPUTS][HIDDENS];
	float weights_hh[HLAYERS-1][HIDDENS][HIDDENS];
	float weights_ho[HIDDENS][OUTPUTS];
};

struct Node{
	float netin;
	float activation;
	float dEdnet;
};

struct calculationnet{
	Node input[INPUTS];
	Node hiddens[HLAYERS][HIDDENS];
	Node output[OUTPUTS];
	
	float dweights_ih[INPUTS][HIDDENS];
	float dweights_hh[HLAYERS-1][HIDDENS][HIDDENS];
	float dweights_ho[HIDDENS][OUTPUTS];
};
*/

float actfunc(float n){
	return 1/(1 + pow(2.71, -n));
}

void calculate(neuralnet &net, calculationnet &cnet){
	
	//replacing the old define-constants
	int INPUTS  = net.ninputs;
	int HLAYERS = net.nhlayers;
	int HIDDENS = net.nhiddens;
	int OUTPUTS = net.noutputs;
	
	//calculate in-hidden
	for (int hid=0; hid<HIDDENS; hid++){
		cnet.hiddens[0][hid].netin = 0;
		for (int in=0; in<INPUTS; in++){
			cnet.hiddens[0][hid].netin += cnet.input[in].activation * net.weights_ih[in][hid];
		}
		cnet.hiddens[0][hid].activation = actfunc(cnet.hiddens[0][hid].netin);
	}
	//calculate hidden-hidden for all layers
	for (int layer=0; layer<HLAYERS-1; layer++){
		
		for (int hout=0; hout<HIDDENS; hout++){
			cnet.hiddens[layer+1][hout].netin = 0;
			for (int hin=0; hin<HIDDENS; hin++){
				
				cnet.hiddens[layer+1][hout].netin += cnet.hiddens[layer][hin].activation * net.weights_hh[layer][hin][hout];
				
			}
			cnet.hiddens[layer+1][hout].activation = actfunc(cnet.hiddens[layer+1][hout].netin);
		}
		
	}
	//calculate hidden-out
	for (int out=0; out<OUTPUTS; out++){
		cnet.output[out].netin = 0;
		for (int hid=0; hid<HIDDENS; hid++){
			
			cnet.output[out].netin += cnet.hiddens[HLAYERS-1][hid].activation * net.weights_ho[hid][out];
			
		}
		cnet.output[out].activation = actfunc(cnet.output[out].netin);
	}
}

void backpropagate(neuralnet &net, calculationnet &cnet, float* input, float* exp_out, float learnrate, float momentum){
	
	//replacing the old define-constants
	int INPUTS  = net.ninputs;
	int HLAYERS = net.nhlayers;
	int HIDDENS = net.nhiddens;
	int OUTPUTS = net.noutputs;
	
	for (int ins=0; ins<INPUTS; ins++){
		cnet.input[ins].activation = input[ins];
	}
	calculate(net, cnet);
	//calculate dEdnet for each outputnode
	for (int out=0; out<OUTPUTS; out++){
		float& o = cnet.output[out].activation;
		cnet.output[out].dEdnet = (o - exp_out[out])*o*(1-o);
	}
	//calculate dEdnet for hidden-out
	for (int hid=0; hid<HIDDENS; hid++){
		
		Node &n = cnet.hiddens[HLAYERS-1][hid];
		n.dEdnet = 0;
		
		for (int out=0; out<OUTPUTS; out++){
			
			n.dEdnet += cnet.output[out].dEdnet * net.weights_ho[hid][out];
			
		}
		
		n.dEdnet *= n.activation * (1 - n.activation);
	}
	//calculate dEdnet for hidden-hidden
	for (int layer = HLAYERS-2; layer>=0; layer--){
		
		for (int hin=0; hin<HIDDENS; hin++){
			
			Node &n = cnet.hiddens[layer][hin];
			n.dEdnet = 0;
			
			for (int hout=0; hout<HIDDENS; hout++){
				
				n.dEdnet += cnet.hiddens[layer+1][hout].dEdnet * net.weights_hh[layer][hin][hout];
				
			}
			
			n.dEdnet *= n.activation * (1 - n.activation);
		}
		
	}
	//no need to calculate dEdnet for the inputs
	
	//modify hidden-out weights
	for (int hin=0; hin<HIDDENS; hin++){
		
		for (int out=0; out<OUTPUTS; out++){
			
			float yps = cnet.output[out].dEdnet;
			float oi = cnet.hiddens[HLAYERS-1][hin].activation;
			float dW = yps * oi * learnrate * (1-momentum)   +   momentum * cnet.dweights_ho[hin][out];
			
			net.weights_ho[hin][out] -= dW;
			
			cnet.dweights_ho[hin][out] = dW;
		}
		
	}
	//modify hidden-hidden weights
	for (int layer=0; layer<HLAYERS-1; layer++){
		
		for (int hin=0; hin<HIDDENS; hin++){
			
			for (int hout=0; hout<HIDDENS; hout++){
				
				float yps = cnet.hiddens[layer+1][hout].dEdnet;
				float oi = cnet.hiddens[layer][hin].activation;
				float dW = yps * oi * learnrate * (1-momentum)   +   momentum * cnet.dweights_hh[layer][hin][hout];
				
				net.weights_hh[layer][hin][hout] -= dW;
				
				cnet.dweights_hh[layer][hin][hout] = dW;
			}
			
		}
		
	}
	//modify input-hidden weights
	for (int in = 0; in<INPUTS; in++){
		
		for (int hout = 0; hout<HIDDENS; hout++){
			
			float yps = cnet.hiddens[0][hout].dEdnet;
			float oi = cnet.input[in].activation;
			float dW = yps * oi * learnrate * (1-momentum)   +   momentum * cnet.dweights_ih[in][hout];
			
			net.weights_ih[in][hout] -= dW;
			
			cnet.dweights_ih[in][hout] = dW;
		}
		
	}
}

float genumb(float range){
	return (((float)(rand()%1000))*range*2)/1000 - range;
}

void randomize(neuralnet &net, float range){
	
	//replacing the old define-constants
	int INPUTS  = net.ninputs;
	int HLAYERS = net.nhlayers;
	int HIDDENS = net.nhiddens;
	int OUTPUTS = net.noutputs;
	
	for (int in=0; in<INPUTS; in++){
		for (int hout=0; hout<HIDDENS; hout++){
			net.weights_ih[in][hout] = genumb(range);
		}
	}
	for (int layer = 0; layer<HLAYERS-1; layer++){
		
		for (int hin=0; hin<HIDDENS; hin++){
			for (int hout=0; hout<HIDDENS; hout++){
				net.weights_hh[layer][hin][hout] = genumb(range);
			}
		}
		
	}
	for (int hin=0; hin<HIDDENS; hin++){
		for (int out=0; out<OUTPUTS; out++){
			net.weights_ho[hin][out] = genumb(range);
		}
	}
}


void init(neuralnet &net, int _inputs, int _hlayers, int _hiddens, int _outputs){
	net.ninputs = _inputs;
	net.nhlayers = _hlayers;
	net.nhiddens = _hiddens;
	net.noutputs = _outputs;
	
	//weights_ih
	net.weights_ih = new float*[_inputs];
	for (int i=0; i<_inputs; i++){
		net.weights_ih[i] = new float[_hiddens];
	}
	
	//weights_hh
	net.weights_hh = new float**[_hlayers-1];
	for (int i=0; i<_hlayers-1; i++){
		net.weights_hh[i] = new float*[_hiddens];
		for (int j=0; j<_hiddens; j++){
			net.weights_hh[i][j] = new float[_hiddens];
		}
	}
	
	//weights_ho
	net.weights_ho = new float*[_hiddens];
	for (int i=0; i<_hiddens; i++){
		net.weights_ho[i] = new float[_outputs];
	}
}

void free(neuralnet &net){
	//weights_ih
	for (int i=0; i<net.ninputs; i++){
		delete net.weights_ih[i];
	}
	delete net.weights_ih;
	
	//weights_hh
	for (int i=0; i<net.nhlayers; i++){
		for (int j=0; j<net.nhiddens; j++){
			delete net.weights_hh[i][j];
		}
		delete net.weights_hh[i];
	}
	delete net.weights_hh;
	
	//weights_ho
	for (int i=0; i<net.nhiddens; i++){
		delete net.weights_ho[i];
	}
	delete net.weights_ho;
}

void init(calculationnet &cnet, int _inputs, int _hlayers, int _hiddens, int _outputs){
	cnet.ninputs = _inputs;
	cnet.nhlayers = _hlayers;
	cnet.nhiddens = _hiddens;
	cnet.noutputs = _outputs;
	
	//input
	cnet.input = new Node[_inputs];
	
	//hiddens
	cnet.hiddens = new Node*[_hlayers];
	for (int i=0; i<_hlayers; i++){
		cnet.hiddens[i] = new Node[_hiddens];
	}
	
	//output
	cnet.output = new Node[_outputs];
	
	//dweights_ih
	cnet.dweights_ih = new float*[_inputs];
	for (int i=0; i<_inputs; i++){
		cnet.dweights_ih[i] = new float[_hiddens];
	}
	
	//dweights_hh
	cnet.dweights_hh = new float**[_hlayers-1];
	for (int i=0; i<_hlayers-1; i++){
		cnet.dweights_hh[i] = new float*[_hiddens];
		for (int j=0; j<_hiddens; j++){
			cnet.dweights_hh[i][j] = new float[_hiddens];
		}
	}
	
	//dweights_ho
	cnet.dweights_ho = new float*[_hiddens];
	for (int i=0; i<_hiddens; i++){
		cnet.dweights_ho[i] = new float[_outputs];
	}
}

void free(calculationnet &cnet){
	int inputs = cnet.ninputs;
	int hlayers = cnet.nhlayers;
	int hiddens = cnet.nhiddens;
	int outputs = cnet.noutputs;
	
	//input
	delete cnet.input;
	
	//hiddens
	for (int i=0; i<hlayers; i++){
		delete cnet.hiddens[i];
	}
	delete cnet.hiddens;
	
	//output
	delete cnet.output;
	
	//dweights_ih
	for (int i=0; i<inputs; i++){
		delete cnet.dweights_ih[i];
	}
	delete cnet.dweights_ih;
	
	//dweights_hh
	for (int i=0; i<hlayers-1; i++){
		for (int j=0; j<hiddens; j++){
			delete cnet.dweights_hh[i][j];
		}
		delete cnet.dweights_hh[i];
	}
	delete cnet.dweights_hh;
	
	//dweights_ho
	for (int i=0; i<hiddens; i++){
		delete cnet.dweights_ho[i];
	}
	delete cnet.dweights_ho;
}

bool save(neuralnet &net, string filename){
return false;
	unsigned char *ptr = (unsigned char*)&net;
	int fd = open(filename.c_str(), O_CREAT | O_RDWR, 0666);
	if (fd<0){
		cout<<"unable to open save file: "<<filename<<endl<<"	errno: "<<errno<<endl;
		return false;
	}
	int n = write(fd, ptr, sizeof(neuralnet));
	close(fd);
	if (n != sizeof(neuralnet)){
		cout<<"unable to save neural net\n";
		return false;
	}
	return true;
}

bool load(neuralnet &net, string filename){
return false;
	unsigned char *ptr = (unsigned char*)&net;
	int fd = open(filename.c_str(), O_RDWR);
	if (fd<0){
		cout<<"unable to open load file: "<<filename<<endl<<"	errno: "<<errno<<endl;
		return false;
	}
	int n = read(fd, ptr, sizeof(neuralnet));
	close(fd);
	if (n != sizeof(neuralnet)){
		cout<<"unable to load neural net\n";
		return false;
	}
	return true;
}








