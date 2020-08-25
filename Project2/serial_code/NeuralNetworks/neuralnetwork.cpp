#include "neuralnetwork.h"
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

using namespace arma;

NeuralNetwork::NeuralNetwork(System* system, double eta, int numberOfInputs, int numberOfHidden) :
          Network(system){
          assert(eta > 0);
          m_eta = eta;

          // Setting to 0 so that the first update of t will be t=tprev+f=tprev.
          m_gradPrevBiasA.zeros(numberOfInputs);
          m_gradPrevBiasB.zeros(numberOfHidden);
          m_gradPrevWeights.zeros(numberOfInputs*numberOfHidden);
}

vec NeuralNetwork::computeBiasAgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_system->getWaveFunction()->getParameters()[1];
    int nx = m_system->getNumberOfInputs();

    vec x = getPositions();
    vec a = getBiasA();
    vec agradient(nx);

    for (int m = 0; m < nx; m++){
      agradient(m) = (x(m) - a(m))/(gibbs*sigma2);
    }

    return agradient;
}

vec NeuralNetwork::computeBiasBgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_system->getWaveFunction()->getParameters()[1];
    int nh = m_system->getNumberOfHidden();

    vec x = getPositions();
    vec b = getBiasB();
    mat W = getWeigths();

    vec Q = b + (((1.0/sigma2)*x).t()*W).t();
    vec bgradient(nh);

    for (int j = 0; j < nh; j++) {
        bgradient(j) = 1.0/(gibbs*(1.0+exp(-Q(j))));
    }

    return bgradient;
}


vec NeuralNetwork::computeWeightsgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double gibbs = m_system->getWaveFunction()->getParameters()[1];
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec x = getPositions();
    vec b = getBiasB();
    mat W = getWeigths();

    vec Q = b + (((1.0/sigma2)*x).t()*W).t();
    vec wgradient(nx*nh);

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nh; j++) {
          wgradient(i*nh + j) = x(i)/(gibbs*sigma2*(1.0+exp(-Q(j))));
      }
    }

    return wgradient;
}



void NeuralNetwork::GradientDescent(vec agrad, vec bgrad, vec wgrad){
  // Compute new parameters
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  for (int i = 0; i < nx; i++){
      m_biasA(i) = m_biasA(i) - m_eta*agrad(i);
  }

  for (int j = 0; j < nh; j++){
      m_biasB(j) = m_biasB(j) - m_eta*bgrad(j);
  }

  for (int i = 0; i < nx; i++){
      for (int j = 0; j < nh; j++){
          m_weights(i,j) = m_weights(i,j) - m_eta*wgrad(i*nh + j);
      }
  }

}


void NeuralNetwork::StochasticGradientDescent(vec agrad, vec bgrad, vec wgrad) {
  /* Calculate the learning rate: gamma = a/(t_i + A)
   * Calculate the new t=max(0, tprev + f), with f responsible for altering the learning rate
   * according to changes in the gradient:
   *
   * If we passed a minimum then the negative product between
   * the current and previous gradient, gradProduct, will be positive. Then f in [0, fmax]
   * and t=tprev + f. This causes an increase in t, causing the learning rate to decrease.
   *
   * If the gradient in two conecutive steps point in the same direction, gradProduct is
   * negative, and f in [fmin, 0]. This reduces t, causing the learning rate to increase.
   * If f is so small that f<-tprev, then t=0 and the learning rate is gamma=a/A, its maximum.
   *
   * To clarify: t_i is here based on the product of gradients from iteration i and i-1.
   * Note: this implementation only really uses the t0 user setting. t1 is here updated according
   * to the algorithm, not the user's choice. m_tprev does not need to be a variable in this
   * implementation. Have still kept both t0 and t1 as input parameters for now as it's mentioned in
   * the algorithm description, in case of future changes. */

   int nx = m_system->getNumberOfInputs();
   int nh = m_system->getNumberOfHidden();

   double gradProduct = -dot(agrad, m_gradPrevBiasA) - dot(bgrad, m_gradPrevBiasB) - dot(wgrad, m_gradPrevWeights);

   double f = m_fmin + (m_fmax - m_fmin)/(1 - (m_fmax/m_fmin)*exp(-gradProduct/m_asgdOmega));
   double tnext = m_tprev + f;
   // Update m_t
   m_t = 0.0;
   if (0.0 < tnext) m_t=tnext;
   // Compute the learning rate
   double eta = m_a/(m_t+m_A);


   // Compute new parameters
   for (int i = 0; i < nx; i++) {
     m_biasA(i) = m_biasA(i) - eta*agrad(i);
   }

   for (int j = 0; j < nh; j++) {
     m_biasB(j) = m_biasB(j) - eta*bgrad(j);
   }

   for (int i = 0; i < nx; i++) {
     for (int j = 0; j < nh; j++) {
       m_weights(i,j) = m_weights(i,j) - eta*wgrad(i*nh + j);
       }
   }

   m_gradPrevBiasA = agrad;
   m_gradPrevBiasB = bgrad;
   m_gradPrevWeights = wgrad;

   m_tprev = m_t;

}


void NeuralNetwork::setPositions(vec &positions) {
    assert(positions.size() == m_system->getNumberOfInputs());
    m_positions = positions;
}

void NeuralNetwork::adjustPositions(double change, int dimension, int input) {
    int n = m_system->getNumberOfParticles();
    m_positions(input*n + dimension) += change;
}

void NeuralNetwork::setWeights(mat &weights) {
    m_weights = weights;
}


void NeuralNetwork::setBiasA(vec &biasA) {
    m_biasA = biasA;
}

void NeuralNetwork::setBiasB(vec &biasB) {
    m_biasB = biasB;
}
