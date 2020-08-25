#include <cmath>
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../NeuralNetworks/network.h"

#include "neuralquantumstate.h"

using namespace arma;
using namespace std;

NeuralQuantumState::NeuralQuantumState(System* system, double sigma, double gibbs) :
        WaveFunction(system) {
    assert(sigma >= 0);
    assert(gibbs >= 0);
    m_numberOfParameters = 2;
    m_parameters.reserve(2);
    m_parameters.push_back(sigma);
    m_parameters.push_back(gibbs);
}

double NeuralQuantumState::evaluate() {
    // Here we compute the wave function.
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_parameters[1];
    double psi1 = 0, psi2 = 1;
    double psi;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec x = m_system->getNetwork()->getPositions();
    vec a = m_system->getNetwork()->getBiasA();
    vec b = m_system->getNetwork()->getBiasB();
    mat W = m_system->getNetwork()->getWeigths();

    vec Q = b + (((1.0/sigma2)*x).t()*W).t();

    for (int i = 0; i < nx; i++){
       psi1 += (x(i) - a(i)) * (x(i) - a(i));
    }

    psi1 = exp(-psi1/(2*sigma2));

    for (int j = 0; j < nh; j++) {
        psi2 *= (1 + exp(Q(j)));
    }

    if (gibbs == 2){
      psi = sqrt(psi1*psi2);
    }
    else{
      psi = psi1*psi2;
    }

    return psi;
}

vec NeuralQuantumState::computeFirstDerivative() {
    // Here we compute the first derivative
    //of the wave function with respect to the visible nodes

    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_parameters[1];
    double temp;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec psi1(nx);

    vec x = m_system->getNetwork()->getPositions();
    vec a = m_system->getNetwork()->getBiasA();
    vec b = m_system->getNetwork()->getBiasB();
    mat W = m_system->getNetwork()->getWeigths();

    vec Q = b + (((1.0/sigma2)*x).t()*W).t();

    for (int i = 0; i < nx; i++){
      temp = 0;
      for (int j = 0; j < nh; j++){
        temp += W(i,j)/(1.0+exp(-Q(j)));
      }
      psi1(i) = -(x(i) - a(i))/(gibbs*sigma2) + temp/(gibbs*sigma2);

    }

    return psi1;
}

vec NeuralQuantumState::computeDoubleDerivative() {
    // Here we compute the double derivative of the wavefunction
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_parameters[1];
    double temp;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec psi2(nx);
    vec x = m_system->getNetwork()->getPositions();
    vec b = m_system->getNetwork()->getBiasB();
    mat W = m_system->getNetwork()->getWeigths();

    vec Q = b + (((1.0/sigma2)*x).t()*W).t();

    for (int i = 0; i < nx; i++){
      temp = 0;
      for (int j = 0; j < nh; j++){
        temp += W(i,j)*W(i,j)*exp(-Q(j))/((1.0+exp(-Q(j)))*(1.0+exp(-Q(j))));
      }
      psi2(i) = -1.0/(gibbs*sigma2) + temp/(gibbs*sigma2*sigma2);

    }

    return psi2;
}
