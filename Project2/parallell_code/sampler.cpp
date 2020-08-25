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
#include "Block/blocker.h"
#include <mpi.h>

using std::cout;
using std::endl;
using namespace arma;

ofstream ofile;

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
    m_localacceptedSteps = counter;
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

  m_globalaDelta.zeros(nx);
  m_globalEaDelta.zeros(nx);

  m_localaDelta.zeros(nx);
  m_localEaDelta.zeros(nx);
  m_agrad.zeros(nx);

  m_globalbDelta.zeros(nh);
  m_globalEbDelta.zeros(nh);
  m_localbDelta.zeros(nh);
  m_localEbDelta.zeros(nh);
  m_bgrad.zeros(nh);

  m_localwDelta.zeros(nx*nh);
  m_localEwDelta.zeros(nx*nh);
  m_globalwDelta.zeros(nx*nh);
  m_globalEwDelta.zeros(nx*nh);
  m_wgrad.zeros(nx*nh);


}


void Sampler::sample(int step) {
    // Making sure the sampling variable(s) are initialized at the first step.
    m_stepNumber = step - 1;
    if (m_stepNumber == 0) {
      m_localcumulativeEnergy = 0;
      m_globalcumulativeEnergy = 0;

      m_localcumulativeEnergy2 = 0;
      m_globalcumulativeEnergy2 = 0;

      m_globalacceptedSteps = 0;
      m_localacceptedSteps = 0;

      m_DeltaPsi = 0;
      m_DerivativePsiE = 0;
      setGradients();
    }

    double localEnergy = m_system->getHamiltonian()->
                     computeLocalEnergy(m_system->getNetwork());

    vec temp_aDelta = m_system->getNetwork()->computeBiasAgradients();
    vec temp_bDelta = m_system->getNetwork()->computeBiasBgradients();
    vec temp_wDelta = m_system->getNetwork()->computeWeightsgradients();

    m_localcumulativeEnergy  += localEnergy;
    m_localcumulativeEnergy2  += localEnergy*localEnergy;

    m_localaDelta += temp_aDelta;
    m_localbDelta += temp_bDelta;
    m_localwDelta += temp_wDelta;

    m_localEaDelta += temp_aDelta*localEnergy;
    m_localEbDelta += temp_bDelta*localEnergy;
    m_localEwDelta += temp_wDelta*localEnergy;

    m_stepNumber++;
}

void Sampler::printOutputToTerminal(int numberOfProcesses) {

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
      cout << " Number of Processes : " << numberOfProcesses << endl;
      cout << " Number of Monte Carlo cycles per process : 2^" << std::log2(ms) << endl;
      cout << " Total number of Monte Carlo cycles : 2^" << std::log2(ms*numberOfProcesses) << endl;
      cout << " Number of equilibration steps per rank : 2^" << std::log2(ms*ef) << endl;
      cout << endl;
      cout << "  -- Wave function parameters -- " << endl;
      cout << " Number of parameters : " << p << endl;
      for (int i=0; i < p; i++) {
          cout << " Parameter " << i+1 << " : " << pam(i) << endl;
      }
      cout << endl;
      cout << "  -- Results -- " << endl;
      cout << " Energy : " << m_globalcumulativeEnergy << endl;
      cout << " Variance : " << m_globalvariance << endl;
      cout << " Local time : " << m_localTime << endl; 
      cout << " Total time : " << m_globalTime << endl;
      if (m_system->getGibbsSampling() == false){
        cout << " # Accepted Steps : " << m_globalacceptedSteps << endl;
      }
      cout << endl;
  }
}

void Sampler::computeAverages(double localTime, int numberOfProcesses) {
    /* Compute the averages of the sampled quantities.
     */

    // Initialisers
    int     nx = m_system->getNumberOfInputs();
    int     nh = m_system->getNumberOfHidden();

    double norm = 1.0/((double) (m_MCcycles*numberOfProcesses));     // divided by  number of cycles
    m_localTime = localTime; 
    m_localcumulativeEnergy = m_localcumulativeEnergy*norm;
    m_localcumulativeEnergy2 = m_localcumulativeEnergy2*norm;
    // should implement blocking in c++. These two quantities overestimate themselves.
    m_localvariance = (m_localcumulativeEnergy2 - m_localcumulativeEnergy*m_localcumulativeEnergy)*norm;
    m_localSTD = sqrt(m_localvariance);


    MPI_Reduce(&m_localvariance,          &m_globalvariance,          1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&m_localSTD,               &m_globalSTD,               1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&m_localcumulativeEnergy2, &m_globalcumulativeEnergy2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&m_localcumulativeEnergy,  &m_globalcumulativeEnergy,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    m_localaDelta *= norm;
    m_localbDelta *= norm;
    m_localwDelta *= norm;

    m_localEaDelta *= norm;
    m_localEbDelta *= norm;
    m_localEwDelta *= norm;

    MPI_Allreduce(m_localaDelta.memptr(), m_globalaDelta.memptr(), nx,    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localbDelta.memptr(), m_globalbDelta.memptr(), nh,    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localwDelta.memptr(), m_globalwDelta.memptr(), nx*nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(m_localEaDelta.memptr(), m_globalEaDelta.memptr(), nx,    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localEbDelta.memptr(), m_globalEbDelta.memptr(), nh,    MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localEwDelta.memptr(), m_globalEwDelta.memptr(), nx*nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Compute gradients
    m_agrad = 2*(m_globalEaDelta - m_globalcumulativeEnergy*m_globalaDelta);
    m_bgrad = 2*(m_globalEbDelta - m_globalcumulativeEnergy*m_globalbDelta);
    m_wgrad = 2*(m_globalEwDelta - m_globalcumulativeEnergy*m_globalwDelta);


    // Optimizer parameters (choose either stochastic gradient descent (SGD) or adaptive SGD (ASGD))
    if (m_system->getOptimizer()){
      m_system->getNetwork()->StochasticGradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    else{
      m_system->getNetwork()->GradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    MPI_Reduce(&m_localacceptedSteps, &m_globalacceptedSteps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime,            &m_globalTime,          1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    m_globalacceptedSteps *= norm;
}


void Sampler::Blocking(int MCcycles){
  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles
  double Energy = m_localcumulativeEnergy*norm;
  m_Blocking(MCcycles-1) = Energy;
}

void Sampler::Energies(int OptCycle, int totOptCycles){
  if (OptCycle==0){
    string file;
    if (m_system->getImportanceSampling()){
        file = "Python/Results/Importance_Sampling/data.dat";
    }

    else if (m_system->getGibbsSampling()){
        file = "Python/Results/Gibbs/data.dat";
    }
    else{
        file = "Python/Results/Brute_Force/data.dat";
    }
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycle
    ofile << setw(15) << setprecision(8) << "Mean"; // Mean energy
    ofile << setw(15) << setprecision(8) << "mse_mean"; // mse energy
    ofile << setw(15) << setprecision(8) << "stdErr"; // std error
    ofile << setw(15) << setprecision(8) << "mse_stdErr" << endl; // mse std error
  }
  Blocker block(m_Blocking);
  ofile << setw(15) << setprecision(8) << OptCycle;
  ofile << setw(15) << setprecision(8) << block.mean;
  ofile << setw(15) << setprecision(8) << block.mse_mean;
  ofile << setw(15) << setprecision(8) << block.stdErr;
  ofile << setw(15) << setprecision(8) << block.mse_stdErr << endl; 

  m_Energies(OptCycle) = m_globalcumulativeEnergy;
  if (totOptCycles == (OptCycle+1)){
     ofile.close();
  } 
}


void Sampler::WriteBlockingtoFile(ofstream& ofile){

  for (int i = 0; i < m_MCcycles; i++){
    ofile << setw(15) << setprecision(8) << m_Blocking(i) << endl; // Mean energy
  }

}
