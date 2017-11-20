#include <iostream>
#include <cstring>

#include "neuralnetwork.hpp"

using namespace std;

void handleArguments(int narg, char** args);
void printhelp();
void loadTestData(char* filename);
void loadNetwork(char* filename);

char defaultvalids[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.- ";

bool action_training = false;
bool action_generating = false;
char *testdatapath = NULL;
char *loadpath = NULL;
char *savepath = NULL;
int cyclecount = 0;
int savecycle = 0;
char *valids = defaultvalids;
int validcount = sizeof(defaultvalids);
int hiddenlayers, hiddencount;
int batchsize = 25;

int currentcycle = 0;

"ACTIONS:\n"<<
"  train           -Train network\n"<<
"  generate        -Generate text\n\n"<<
"ARGUMENTS\n"<<
"  -h              show this menu\n"<<
"  -f [filename]   select file for testdata\n"<<
"  -l [filename]   select file to load network from, overrides -d\n"<<
"  -s [filename]   select file to save network to\n"<<
"  -c [number]     cycles to do, either train or generate, 0 for infinite, default is infinite\n"<<
"  -n [number]     if savefile is given, save network every n cycles, only when training, 0 is never, default is never\n"<<
"  -v [string]     valid characters, coherent string of characters to be used as inputs, default is a-z, A-Z, 0-9 and '., '\n"<<
"                  MUST at least be same count as used with network previously, if loading network from file\n"<<
"  -d [dimentions] specify 'dimentions' of new network, in format '[hidden layers]:[hiddens]'\n"<<
"  -b [number]     the batch size to be used when training, default is 25\n";

int main(int narg, char** args){
	handleArguments(narg, args);
	return 0;
}


void handleArguments(int narg, char** args){
	int i=1;
	while (i<narg){
		
		if (strcmp(args[i], "train")==0){
			action_training = true;
		}else
		if (strcmp(args[i], "generate")==0){
			action_generating = true;
		}else
		if (strcmp(args[i], "-h")==0){
			printhelp();
		}else
		if (strcmp(args[i], "-f")==0){
			i++;
			testdatapath = args[i];
		}else
		if (strcmp(args[i], "-l")==0){
			i++;
			loadpath = args[i];
		}else
		if (strcmp(args[i], "-s")==0){
			i++;
			if (i!=narg){
				savepath = args[i]
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
				valids = args[i];
			}
		}else
		if (strcmp(args[i], "-d")==0){
			i++;
			char strbuff[16];
			int sizes[2];
			int k=0;
			int j=0;
			while (args[i][k-1] != '\0' && k<16 && j<2){
				if (args[i][k] != ':' && args[i][k] != '\0'){
					strbuff[k] = args[i][k];
					k++;
				}else{
					strbuff[k] = '\0';
					sizes[j] = atoi(strbuff);
					k=0;
					j++;
				}
			}
			if (j==2){
				cout<<"dimentions: "<<sizes[0]<<" "<<sizes[1]<<"\n";
				hiddenlayers = sizes[0];
				hiddencount = sizes[1];
			}else{
				cout<<"Unable to parse "<<args[i]<<"\n";
			}
		}else
		if (strcmp(args[i], "-b")==0){
			i++;
			batchsize = atoi(args[i]);
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
"  -l [filename]   select file to load network from, overrides -d\n"<<
"  -s [filename]   select file to save network to\n"<<
"  -c [number]     cycles to do, either train or generate, 0 for infinite, default is infinite\n"<<
"  -n [number]     if savefile is given, save network every n cycles, only when training, 0 is never, default is never\n"<<
"  -v [string]     valid characters, coherent string of characters to be used as inputs, default is a-z, A-Z, 0-9 and '., '\n"<<
"                  MUST at least be same count as used with network previously, if loading network from file\n"<<
"  -d [dimentions] specify 'dimentions' of new network, in format '[characters in]:[hidden layers]:[hiddens]'\n"<<
"  -b [number]     the batch size to be used when training, default is 25\n";
}

