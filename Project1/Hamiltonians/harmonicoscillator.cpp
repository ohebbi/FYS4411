#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;
using namespace arma;

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    assert(omega > 0);
    m_omega  = omega;
}

/* In this class, we are computing the energies. Note that
 * when using numerical differentiation, the computation of the kinetic
 * energy becomes the same for all Hamiltonians, and thus the code for
 * doing this should be moved up to the super-class, Hamiltonian.
 * (Great tip that should be implemented in the future)
 */

double HarmonicOscillator::computePotentialEnergy(std::vector<Particle*> particles) {
    // Here we compute the potential energy.

    double gamma = m_system->getWaveFunction()->getParameters()[2];

    int N = m_system->getNumberOfParticles(); // Number of Particles
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    double r2;
    double psi = m_system->getWaveFunction()->evaluate(particles); // psi(r)
    double potentialenergy;

    r2 = 0;
    for (int i = 0; i < N; i++){
      for (int j = 0; j < Dim; j++){
        if (j == 2){
            r2 += gamma*gamma*m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
        }
        else{
            r2 += m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
        }
      }
    }

    potentialenergy = 0.5*r2;
    return potentialenergy;
}

double HarmonicOscillator::computeRepulsiveInteraction(std::vector<Particle*> particles) {
    // Here we compute the repulsive interaction.

    double alpha = m_system->getWaveFunction()->getParameters()[0];
    double gamma = m_system->getWaveFunction()->getParameters()[2];
    double a = m_system->getWaveFunction()->getParameters()[3];

    int N = m_system->getNumberOfParticles(); // Number of Particles
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    double ri, rj, temp_i, temp_j, diff;
    double psi = m_system->getWaveFunction()->evaluate(particles); // psi(r)
    double potentialenergy;
    double infty = 0.0;

    potentialenergy = 0;
    for (int i = 0; i < N; i++){
      for (int j = i+1; j < N; j++){
      rj = 0;
      ri = 0;
        for (int k = 0; k < Dim; k++){
          temp_i = m_system->getParticles()[i]->getPosition()[k];
          temp_j = m_system->getParticles()[j]->getPosition()[k];
          ri += temp_i*temp_i; // x^2 + y^2 + z^2
          rj += temp_j*temp_j; // x^2 + y^2 + z^2
          }
        ri = sqrt(ri);
        rj = sqrt(rj);
        diff = fabs(ri - rj);

        if (diff <= a){
          potentialenergy += infty;
        }
      }
    }
    return potentialenergy;
}


double HarmonicOscillator::computeLocalEnergyAnalytical(std::vector<Particle*> particles) {
    // Here we compute the analytical local energy
    double analytical_E_L;
    double potentialenergy = computePotentialEnergy(particles);
    double analytical_kineticenergy = m_system->getWaveFunction()->computeDoubleDerivative(particles);
    double repulsiveInteraction = 0;

    if (m_system->getRepulsivePotential()){
        repulsiveInteraction = computeRepulsiveInteraction(particles);
        analytical_kineticenergy += m_system->getWaveFunction()->computeDoubleDerivativeInteraction(particles);
    }
    analytical_E_L = analytical_kineticenergy + potentialenergy + repulsiveInteraction;
    return analytical_E_L;
}


double HarmonicOscillator::computeLocalEnergyNumerical(std::vector<Particle*> particles) {
    // Here we compute the local energy with a numerical scheme.

    double alpha = m_system->getWaveFunction()->getParameters()[0];
    int N = m_system->getNumberOfParticles(); // Number of Particles
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    double r2, temp;
    double psi = m_system->getWaveFunction()->evaluate(particles); // psi(r)
    double numerical_kineticenergy, potentialenergy, numerical_E_L;

    potentialenergy = computePotentialEnergy(particles);
    numerical_kineticenergy = m_system->getWaveFunction()->computeDoubleNumericalDerivative(particles);
    numerical_E_L = numerical_kineticenergy + potentialenergy;

    return numerical_E_L;
}


std::vector<double> HarmonicOscillator::computeQuantumForce(std::vector<Particle*> particles, int i) {
   // Here we compute the quantum force used in importance sampling

   int Dim = m_system->getNumberOfDimensions(); // The Dimension
   int N = m_system->getNumberOfParticles(); // Number of Particles
   double alpha = m_system->getWaveFunction()->getParameters()[0];
   double beta = m_system->getWaveFunction()->getParameters()[1];

   std::vector<double> force;
     for (int j = 0; j < Dim; j++){
       if (j==2){
         force.push_back(-4*alpha*beta*m_system->getParticles()[i]->getPosition()[j]);
        }
       else{
         force.push_back(-4*alpha*m_system->getParticles()[i]->getPosition()[j]);
       }
     }
  return force;
  }
