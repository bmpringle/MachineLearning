#include "NeuralNetwork.h"

#include "exprtk/exprtk.hpp"
#include <iostream>

void initExprtk(exprtk::expression<double>*& expr, exprtk::symbol_table<double>*& s_table, exprtk::parser<double>*& parser, double (*model)(double), double (*error_function)(double, double), double& x, double& t, std::vector<double>* weights) {
    s_table = new exprtk::symbol_table<double>;
    s_table->add_function("model", model);
    s_table->add_function("errorFunction", error_function);
    s_table->add_variable("x", x);
    s_table->add_variable("t", t);

    int i = 1;

    for(double& weight : *weights) {
        s_table->add_variable("w" + std::to_string(i), weight);
        ++i;
    }
    
    expr = new exprtk::expression<double>;
    expr->register_symbol_table(*s_table);
    parser = new exprtk::parser<double>;
    
    if (!parser->compile("errorFunction(model(x), t)", *expr)) {
        throw std::runtime_error(parser->error());
    }
}

double getExprValue(exprtk::expression<double>* expr) {
    return expr->value();
}

double getExprDerivative(exprtk::expression<double>* expr, std::string respected_variable) {
    return exprtk::derivative(*expr, respected_variable);
}

void cleanupExprtk(exprtk::expression<double>*& expr, exprtk::symbol_table<double>*& s_table, exprtk::parser<double>*& parser) {
    delete expr;
    delete s_table;
    delete parser;
}
