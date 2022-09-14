#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

namespace exprtk { //forward-declare these types so we don't have to include the (very large) exprtk header here
    template<class>
    class expression;

    template<class>
    class symbol_table;

    template<class>
    class parser;
}

struct DataPoint {
    DataPoint(double i, double o) : input(i), output(o) {

    }

    double input;
    double output;
};

void initExprtk(exprtk::expression<double>*& expr, exprtk::symbol_table<double>*& s_table, exprtk::parser<double>*& parser, double (*model)(double), double (*error_function)(double, double), double& x, double& t, std::vector<double>* weights);
double getExprValue(exprtk::expression<double>* expr);
double getExprDerivative(exprtk::expression<double>* expr, std::string respected_variable);

template<int C>
class NeuralNetwork {
    public:
        NeuralNetwork(double (*model)(double), double (*error_function)(double, double), std::vector<double>* weights) : weights(weights), prev_weights(*weights), weight_count(weights->size()) {
            initExprtk(expr, s_table, parser, model, error_function, x, t, weights);    
        } 

        void addDataPoints(std::vector<DataPoint> data) {
            for(DataPoint& dp : data) {
                trainingData.push_back(dp);
            }

            prev_avg_loss = getAverageLoss();
        }

        void doTrainingCycle(std::vector<double>& learning_rate) {
            for(int i = 0; i < weights->size(); ++i) {
                double delta_w = 0;

                for(int d = 0; d < trainingData.size(); ++d) {
                    x = trainingData.at(d).input;
                    t = trainingData.at(d).output;
                    delta_w += getExprDerivative(expr, "w" + std::to_string(i + 1));
                }

                delta_w = delta_w / (double)trainingData.size();

                (*weights)[i] -= learning_rate[i] * delta_w;

                double loss_avg = getAverageLoss();

                if(prev_avg_loss < loss_avg) {
                    (*weights)[i] = prev_weights[i];
                    learning_rate[i] /= 2.0;
                    --i;
                }else {
                    prev_weights[i] = (*weights)[i];
                    loss_avg = prev_avg_loss;
                }
            }
        }

        double getAverageLoss() {
            double loss_avg = 0;

            for(int d = 0; d < trainingData.size(); ++d) {
                x = trainingData.at(d).input;
                t = trainingData.at(d).output;

                loss_avg += getExprValue(expr);
            }
            loss_avg = loss_avg / (double)trainingData.size();

            return loss_avg;
        }

    private:
        std::vector<DataPoint> trainingData;

        int weight_count = 0;

        //weights
        std::vector<double>* weights;

        //function pointers
        double (*model)(double);
        double (*error_function)(double, double);

        //exprtk objects (on heap to allow them to be forward-declared pointers)
        exprtk::expression<double>* expr;
        exprtk::symbol_table<double>* s_table;
        exprtk::parser<double>* parser;

        //input/output variables
        double x = 0;
        double t = 0;

        //misc vars
        double prev_avg_loss;
        std::vector<double> prev_weights;
};

#endif