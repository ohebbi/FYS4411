#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"
#include "../NeuralNetworks/network.h"

#include "harmonicoscillator.h"

using namespace arma;
using std::cout;
using std::endl;

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


double HarmonicOscillator::computeLocalEnergy(Network* network) {
    // Here we compute the analytical local energy
    int nx = m_system->getNumberOfInputs();
    double kineticenergy = 0;
    double potentialenergy = 0;
    double totalenergy = 0;

    vec firstder = m_system->getWaveFunction()->computeFirstDerivative();
    vec secondder = m_system->getWaveFunction()->computeDoubleDerivative();
    vec x = network->getPositions();

    // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
    for (int i = 0; i < nx; i++){
      kineticenergy += -firstder(i)*firstder(i) - secondder(i);
      potentialenergy += m_omega*m_omega*x(i)*x(i);
    }

    totalenergy = 0.5*(kineticenergy + potentialenergy);


    // With interaction:
    if (m_system->getRepulsivePotential()) {
        totalenergy += Interaction();
    }

    return totalenergy;
}

double HarmonicOscillator::Interaction() {

  double interactionTerm = 0;
  double rDistance;
  int nx = m_system->getNumberOfInputs();
  int dim = m_system->getNumberOfDimensions();

  vec x = m_system->getNetwork()->getPositions();

  // Loop over each particle
  for (int r = 0; r < nx - dim; r += dim) {
      // Loop over each particle s that particle r hasn't been paired with
      for (int s = (r + dim); s < nx; s += dim) {
          rDistance = 0;
          // Loop over each dimension
          for (int i = 0; i < dim; i++) {
            rDistance += (x(r+i) - x(s+i))*(x(r+i) - x(s+i));
          }
          interactionTerm += 1.0/sqrt(rDistance);
      }
  }
  return interactionTerm;
}
