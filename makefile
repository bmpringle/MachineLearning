default:
	cd $(FOLDER); clang++ -I ../ NeuralNetwork.cpp main.cpp -O3 -g -std=c++17
	cd $(FOLDER); ./a.out