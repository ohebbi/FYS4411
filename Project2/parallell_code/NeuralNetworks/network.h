#pragma once
#include <vector>
#include <armadillo>
using namespace arma;

class Network {
public:
    Network(class System* system);

    virtual vec computeBiasAgradients() = 0;
    virtual vec computeBiasBgradients() = 0;
    virtual vec computeWeightsgradients() = 0;


    virtual void GradientDescent(vec agrad, vec bgrad, vec wgrad) = 0;
    virtual void StochasticGradientDescent(vec agrad, vec bgrad, vec wgrad) = 0;

    virtual void setPositions(vec &positions) = 0;
    virtual void adjustPositions(double change, int dimension, int input) = 0;
    virtual void setWeights(mat &weights) = 0;
    virtual void setBiasA(vec &biasA) = 0;
    virtual void setBiasB(vec &biasB) = 0;

    virtual vec getPositions() = 0;
    virtual mat getWeigths() = 0;
    virtual vec getBiasA() = 0;
    virtual vec getBiasB() = 0;

protected:

  class System* m_system = nullptr;
};
