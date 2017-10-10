#include <iostream>
#include <cstring>

#include "neuralnetwork.hpp"

using namespace std;

void handleArguments(int narg, char** args);
void printhelp();
void loadTestData(char* filename);
void loadNetwork(char* filename);

char defaultvalids[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.- ";

char *valids = defaultvalids;
int validcount = sizeof(defaultvalids);
char *savepath = NULL;
int cyclecount = 0;
int savecycle = 0;

int currentcycle = 0;


int main(int narg, char** args){
	handleArguments(narg, args);
	return 0;
}


void handleArguments(int narg, char** args){
	int i=1;
	while (i<narg){
		
		if (strcmp(args[i], "-h")==0){
			printhelp();
		}else
		if (strcmp(args[i], "-f")==0){
			if (i+1!=narg)
				exit(-1);//loadTestData(args[i+1]);
			i++;
		}else
		if (strcmp(args[i], "-l")==0){
			if (i+1!=narg)
				exit(-1);//loadNetwork(args[i+1]);
			i++;
		}else
		if (strcmp(args[i], "-s")==0){
			i++;
			if (i!=narg){
				savepath = new char[strlen(args[i])+1];
				memcpy(savepath, args[i], strlen(args[i])+1);
			}
		}else
		if (strcmp(args[i], "-c")==0){
			i++;
			if (i!=narg){
				cyclecount = atoi(args[i]);
			}
		}else
		if (strcmp(args[i], "-n")==0){
			i++;
			if (i!=narg){
				savecycle = atoi(args[i]);
			}
		}else
		if (strcmp(args[i], "-v")==0){
			i++;
			if (i!=narg){
				validcount = strlen(args[i]);
				valids = new char[validcount+1];
				memcpy(valids, args[i], validcount+1);
			}
		}
		
		i++;
	}
}

void printhelp(){
	cout<<
"nn [action] [arguments]\n"<<
"ACTIONS:\n"<<
"  train           -Train network\n"<<
"  generate        -Generate text\n\n"<<
"ARGUMENTS\n"<<
"  -h              show this menu\n"<<
"  -f [filename]   select file for testdata\n"<<
"  -l [filename]   select file to load network from\n"<<
"  -s [filename]   select file to save network to\n"<<
"  -c [number]     cycles to do, either train or generate, 0 for infinite, default is infinite\n"<<
"  -n [number]     if savefile is given, save network every n cycles, only when training, 0 is never, default is never\n"<<
"  -v [string]     valid characters, coherent string of characters to be used as inputs, default is a-z, A-Z, 0-9 and '., '\n"<<
"  -d [dimentions] specify 'dimentions' of new network, in format 'characters in:hidden layers:hiddens'\n";
}

