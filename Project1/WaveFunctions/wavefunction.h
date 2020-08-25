#pragma once
#include <vector>


class WaveFunction {
public:
    WaveFunction(class System* system);
    double computeFirstDerivativeCorrelation(double dist);
    double computeDoubleDerivativeCorrelation(double dist);
    int     getNumberOfParameters() { return m_numberOfParameters; }
    std::vector<double> getParameters() { return m_parameters; }
    virtual double evaluate(std::vector<class Particle*> particles) = 0;
    virtual double derivativeWavefunction(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleDerivative(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleNumericalDerivative(std::vector<class Particle*> particles) = 0;
    virtual double computeDoubleDerivativeInteraction(std::vector<class Particle*> particles) = 0;




protected:
    int     m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
    class System* m_system = nullptr;
};
