#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <armadillo>
#include "sampler.h"
#include "system.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
#include "NeuralNetworks/network.h"

using std::cout;
using std::endl;
using namespace arma;

Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::setMCcyles(int effectiveSamplings) {
    m_MCcycles = effectiveSamplings;
}

void Sampler::setacceptedStep(int counter) {
    m_acceptedStep = counter;
}


void Sampler::setEnergies(int Optcycles) {
  m_Energies.zeros(Optcycles);

}

void Sampler::setBlocking(int MCcycles) {
  m_Blocking.zeros(MCcycles);

}

void Sampler::setGradients() {
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  m_aDelta.zeros(nx);
  m_EaDelta.zeros(nx);
  m_agrad.zeros(nx);

  m_bDelta.zeros(nh);
  m_EbDelta.zeros(nh);
  m_bgrad.zeros(nh);

  m_wDelta.zeros(nx*nh);
  m_EwDelta.zeros(nx*nh);
  m_wgrad.zeros(nx*nh);


}


void Sampler::sample(int step) {
    // Making sure the sampling variable(s) are initialized at the first step.
    m_stepNumber = step - 1;
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
        m_cumulativeEnergy2 = 0;
        m_DeltaPsi = 0;
        m_DerivativePsiE = 0;
        setGradients();
    }


    double localEnergy = m_system->getHamiltonian()->
                     computeLocalEnergy(m_system->getNetwork());

    vec temp_aDelta = m_system->getNetwork()->computeBiasAgradients();
    vec temp_bDelta = m_system->getNetwork()->computeBiasBgradients();
    vec temp_wDelta = m_system->getNetwork()->computeWeightsgradients();

    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2  += localEnergy*localEnergy;

    m_aDelta += temp_aDelta;
    m_bDelta += temp_bDelta;
    m_wDelta += temp_wDelta;


    m_EaDelta += temp_aDelta*localEnergy;
    m_EbDelta += temp_bDelta*localEnergy;
    m_EwDelta += temp_wDelta*localEnergy;

    m_stepNumber++;
}

void Sampler::printOutputToTerminal(double total_time) {

    // Initialisers
    int     nx = m_system->getNumberOfInputs();
    int     nh = m_system->getNumberOfHidden();
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    int     p  = 3;
    double  ef = m_system->getEquilibrationFraction();
    vec pam(3); pam(0) = nx; pam(1) = nh; pam(2) = nx*nh;

    if (m_system->getPrintToTerminal()){
      cout << endl;
      cout << "  -- System info -- " << endl;
      cout << " Number of particles  : " << np << endl;
      cout << " Number of dimensions : " << nd << endl;
      cout << " Number of Monte Carlo cycles : 10^" << std::log10(ms) << endl;
      cout << " Number of equilibration steps  : 10^" << std::log10(std::round(ms*ef)) << endl;
      cout << endl;
      cout << "  -- Wave function parameters -- " << endl;
      cout << " Number of parameters : " << p << endl;
      for (int i=0; i < p; i++) {
          cout << " Parameter " << i+1 << " : " << pam(i) << endl;
      }
      cout << endl;
      cout << "  -- Results -- " << endl;
      cout << " Energy : " << m_energy << endl;
      cout << " Variance : " << m_variance << endl;
      cout << " Time : " << total_time << endl;
      if (m_system->getGibbsSampling() == false){
        cout << " # Accepted Steps : " << m_acceptedStep << endl;
      }
      cout << endl;
  }
}

void Sampler::computeAverages(double total_time) {
    /* Compute the averages of the sampled quantities.
     */
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    int N = m_system->getNumberOfParticles();    // Number of Particles
    double norm = 1.0/((double) (m_MCcycles));     // divided by  number of cycles

    m_energy = m_cumulativeEnergy*norm;
    m_cumulativeEnergy2 = m_cumulativeEnergy2*norm;
    m_cumulativeEnergy = m_cumulativeEnergy*norm;
    m_variance = (m_cumulativeEnergy2 - m_cumulativeEnergy*m_cumulativeEnergy)*norm;
    m_STD = sqrt(m_variance);

    m_aDelta *= norm;
    m_bDelta *= norm;
    m_wDelta *= norm;


    m_EaDelta *= norm;
    m_EbDelta *= norm;
    m_EwDelta *= norm;

    // Compute gradients
    m_agrad = 2*(m_EaDelta - m_cumulativeEnergy*m_aDelta);
    m_bgrad = 2*(m_EbDelta - m_cumulativeEnergy*m_bDelta);
    m_wgrad = 2*(m_EwDelta - m_cumulativeEnergy*m_wDelta);


    // Optimizer parameters (choose either stochastic gradient descent (SGD) or adaptive SGD (ASGD))
    if (m_system->getOptimizer()){
      m_system->getNetwork()->StochasticGradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    else{
      m_system->getNetwork()->GradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    m_totalTime = total_time;
    m_acceptedStep *= norm;
}


void Sampler::Blocking(int MCcycles){

  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles
  double Energy = m_cumulativeEnergy*norm;
  m_Blocking(MCcycles-1) = Energy;
}

void Sampler::Energies(int OptCycles){

  m_Energies(OptCycles) = m_energy;
}


void Sampler::WriteBlockingtoFile(ofstream& ofile){

  for (int i = 0; i < m_MCcycles; i++){
    ofile << setw(15) << setprecision(8) << m_Blocking(i) << endl; // Mean energy
  }

}
