#pragma once
#include "network.h"
#include <vector>
#include <armadillo>
using namespace arma;

class NeuralNetwork : public Network {
public:
    NeuralNetwork(class System* system, double eta, int numberOfInputs, int numberOfHidden);

    vec computeBiasAgradients();
    vec computeBiasBgradients();
    vec computeWeightsgradients();


    void GradientDescent(vec agrad, vec bgrad, vec wgrad);
    void StochasticGradientDescent(vec agrad, vec bgrad, vec wgrad);


    void setPositions(vec &positions);
    void adjustPositions(double change, int dimension, int input);
    void setWeights(mat &weights);
    void setBiasA(vec &biasA);
    void setBiasB(vec &biasB);

    vec getPositions() { return m_positions; }
    mat getWeigths() { return m_weights; }
    vec getBiasA() { return m_biasA; }
    vec getBiasB() { return m_biasB; }

private:
    double     m_eta = 0;
    double     m_a = 0.01;
    double     m_A = 1/m_a;
    double     m_asgdOmega = 1.0;
    double     m_fmax = 2.0;
    double     m_fmin = -0.5;
    double     m_t = m_A;
    double     m_tprev = m_A;

    vec m_gradPrevBiasA;
    vec m_gradPrevBiasB;
    vec m_gradPrevWeights;

    mat m_weights;
    vec m_positions;
    vec m_biasA;
    vec m_biasB;
};
