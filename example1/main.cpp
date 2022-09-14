#include <iostream>
#include <string>
#include <math.h>

#include "NeuralNetwork.h"

std::vector<double> weights = {1};

double linearModel(double x) { //x is the input to the model, w is the weight
    return weights[0] * x;
}

double distanceSquaredErrorFunction(double y, double target_y) { //y is the output of the model, target_y is the target data to match
    return (target_y - y) * (target_y - y);
}

double generateDataPoint(double target_w, double x) {
    int sign = rand() % 2;
    if(sign == 0) {
        sign = -1;
    }
    return x * target_w + ((double)(rand() % 1000) / 500.0) * sign;
}

int main() {
    srand(0); //for consistency of data generation

    int dataCount = 10000;
    double target_w = 5;
    std::vector<double> learning_rates = {0.01};
    double training_iterations = 10;

    std::cout << "Enter slope for data generation: " << std::endl;
    std::cin >> target_w;

    std::cout << "Enter amount of training data to generate: " << std::endl;
    std::cin >> dataCount;

    std::cout << "Enter training iterations to run through gradient descent: " << std::endl;
    std::cin >> training_iterations;

    std::cout << "Enter initial learning rate: " << std::endl;
    std::cin >> learning_rates[0];

    std::cout << "Enter intiial weight value: " << std::endl;
    std::cin >> weights[0];

    std::vector<DataPoint> data;

    for(int d = 0; d < dataCount; ++d) {
        double x = (rand() % 1000) / 100.0;
        data.push_back(DataPoint(x, generateDataPoint(target_w, x)));
    }
    
    NeuralNetwork<1> network = NeuralNetwork<1>(linearModel, distanceSquaredErrorFunction, &weights);
    
    network.addDataPoints(data);

    std::cout << "training start:" << std::endl;

    for(int iteration = 0; iteration < training_iterations; ++iteration) {
        network.doTrainingCycle(learning_rates);
        std::cout << "after iteration " << iteration + 1 << ", w = " << weights[0] << " and loss = " << network.getAverageLoss() << std::endl;
    }
}