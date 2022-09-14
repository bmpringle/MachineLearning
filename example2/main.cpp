#include <iostream>
#include <string>
#include <math.h>

#include "NeuralNetwork/NeuralNetwork.h"

std::vector<double> weights = {1, 1, 1};

double quadraticModel(double x) { //x is the input to the model, w1, w2, and w3 are the weights
    return weights[0] * x * x + weights[1] * x + weights[2];
}

double distanceSquaredErrorFunction(double y, double target_y) { //y is the output of the model, target_y is the target data to match
    return (target_y - y) * (target_y - y);
}

double generateDataPoint(double target_w1, double target_w2, double target_w3, double x) {
    int sign = rand() % 2;
    if(sign == 0) {
        sign = -1;
    }
    return (x * x * target_w1 + x * target_w2 + target_w3) + ((double)(rand() % 1000) / 500.0) * sign;
}

int main() {
    srand(0); //for consistency of data generation

    int dataCount = 10000;

    double target_w1 = 5;
    double target_w2 = 5;
    double target_w3 = 5;

    std::vector<double> learning_rates = {0.01, 0.01, 0.01};
    double training_iterations = 10;

    std::cout << "Enter quadratic coefficents for data generation (order ax^2, bx, c): " << std::endl;
    std::cin >> target_w1;
    std::cin >> target_w2;
    std::cin >> target_w3;

    std::cout << "Enter amount of training data to generate: " << std::endl;
    std::cin >> dataCount;

    std::cout << "Enter training iterations to run through gradient descent: " << std::endl;
    std::cin >> training_iterations;

    std::cout << "Enter initial learning rates for the three weights: " << std::endl;
    std::cin >> learning_rates[0];
    std::cin >> learning_rates[1];
    std::cin >> learning_rates[2];

    std::cout << "Enter intiial weight values for the three weights: " << std::endl;
    std::cin >> weights[0];
    std::cin >> weights[1];
    std::cin >> weights[2];

    std::vector<DataPoint> data;

    for(int d = 0; d < dataCount; ++d) {
        double x = (rand() % 1000) / 100.0;
        data.push_back(DataPoint(x, generateDataPoint(target_w1, target_w2, target_w3, x)));
    }
    
    NeuralNetwork network = NeuralNetwork(quadraticModel, distanceSquaredErrorFunction, &weights);
    
    network.addDataPoints(data);

    std::cout << "training start:" << std::endl;

    for(int iteration = 0; iteration < training_iterations; ++iteration) {
        network.doTrainingCycle(learning_rates);
        std::cout << "after iteration " << iteration + 1 << ", w = [" << weights[0] << ", " << weights[1] << ", " << weights[2] << "] and loss = " << network.getAverageLoss() << std::endl;
    }
}