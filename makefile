
all:
	g++ neuralnetwork.cpp main.cpp -o nn.out -std=c++11

example:
	g++ neuralnetwork.cpp example.cpp -o example.out -std=c++11

