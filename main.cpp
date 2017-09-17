#include "neuralnetwork.hpp"
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

int datasize;
char dataset[170000];

int main(){

	int fd = open("testdata.txt", O_RDONLY);
	if (fd<0){
		cout<<"unable to open \"testdata.txt\" file"<<endl;
		exit(-1);
	}
	datasize = read(fd, dataset, 170000);
	close(fd);
	if (datasize<=0){
		cout<<"unable to read from testdata.txt-file"<<endl;
		exit(-1);
	}
	
	cout<<"done reading #"<<datasize<<endl;

	srand(1997);
	neuralnet net;
	calculationnet cnet;
	
	if (!load(net, "alice.nn")){
		randomize(net, 0.5);
	}
	
	float in[286];
	float memory[30];
	float out[256];
	float errsum=0;
	
	while (true){
	
		for (int i=0;i<datasize-1;i++){
			for (int k=0;k<256;k++){
				if (dataset[i] == k){
					in[k] = 1;
				}else{
					in[k] = 0;
				}
			}
		
			for (int k=256;k<286;k++){
				in[k] = cnet.hiddens[HLAYERS-1][k].activation;
			}
		
			for (int k=0; k<256; k++){
				out[k] = (dataset[i+1] == k)?1:0;
			}
		
			backpropagate(net, cnet, in, out);
		
			int biggest=0;
			for (int k=1;k<256;k++){
				errsum += pow( cnet.output[k].activation-out[k], 2.0 );
				if (cnet.output[k].activation > cnet.output[biggest].activation){
					biggest = k;
				}
			}
			
			if (i % 8192 < 512)
				cout<<(char)biggest<<flush;
			if (i % 8192 == 512-1)
				cout<<endl;
			
			if (i % 8192 == 8192-1){
				cout<<"\nSAVING\n";
				save(net, "alice.nn");
				cout<<"error: "<<errsum/8192<<endl<<endl;
				errsum=0;
			}
		}
		cout<<"\nSAVING\n";
		save(net, "alice.nn");
		cout<<"error: "<<errsum/8192<<endl<<endl;
		errsum=0;
	}
	
	return 0;
}

