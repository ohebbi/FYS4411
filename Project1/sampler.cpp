#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <armadillo>
#include "sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
using namespace arma;
using std::cout;
using std::endl;


Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}


void Sampler::setEnergies(int MCcycles) {
  for (int i = 0; i < MCcycles; i++){
    m_Energies.push_back(0);
  }

}


void Sampler::sample(bool acceptedStep, int MCcycles) {
    // Making sure the sampling variable(s) are initialized at the first step.
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
        m_cumulativeEnergy2 = 0;
        m_DeltaPsi = 0;
        m_DerivativePsiE = 0;
    }

    // Numerical derivative
    if (m_system->getNumericalDerivative()){
        double localEnergy = m_system->getHamiltonian()->
                          computeLocalEnergyNumerical(m_system->getParticles());

        m_cumulativeEnergy  += localEnergy;
        m_cumulativeEnergy2  += localEnergy*localEnergy;
        m_stepNumber++;
     }


    else if (m_system->getOneBodyDensity()){
        computeOneBodyDensity();
    }

    else if (m_system->getRepulsivePotential()){
        double localEnergy = m_system->getHamiltonian()->
                         computeLocalEnergyAnalytical(m_system->getParticles());

        double DerPsi     = m_system->getWaveFunction()->
                        derivativeWavefunction(m_system->getParticles());
        m_DeltaPsi += DerPsi;
        m_DerivativePsiE += DerPsi*localEnergy;
        m_cumulativeEnergy  += localEnergy;
        m_cumulativeEnergy2  += localEnergy*localEnergy;
        m_stepNumber++;
    }


    else{
        double localEnergy = m_system->getHamiltonian()->
                         computeLocalEnergyAnalytical(m_system->getParticles());

        m_cumulativeEnergy  += localEnergy;
        m_cumulativeEnergy2  += localEnergy*localEnergy;
        m_stepNumber++;
    }

}

void Sampler::printOutputToTerminal(double total_time, double acceptedStep) {

    // Initialisers
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    int     p  = m_system->getWaveFunction()->getNumberOfParameters();
    double  ef = m_system->getEquilibrationFraction();
    std::vector<double> pa = m_system->getWaveFunction()->getParameters();

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
          cout << " Parameter " << i+1 << " : " << pa.at(i) << endl;
      }
      cout << endl;
      cout << "  -- Results -- " << endl;
      cout << " Energy : " << m_energy << endl;
      cout << " Variance : " << m_variance << endl;
      cout << " Time : " << total_time << endl;
      cout << " # Accepted Steps : " << acceptedStep << endl;
      cout << endl;
  }
}

void Sampler::computeAverages(double total_time, double acceptedStep) {
    /* Compute the averages of the sampled quantities.
     */
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    int N = m_system->getNumberOfParticles(); // Number of Particles
    int MCcycles = m_system->getNumberOfMetropolisSteps(); // Number of Monte Carlo steps
    double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles

    m_energy = m_cumulativeEnergy *norm;
    m_cumulativeEnergy2 = m_cumulativeEnergy2 *norm;
    m_cumulativeEnergy = m_cumulativeEnergy *norm;
    m_variance = (m_cumulativeEnergy2 - m_cumulativeEnergy*m_cumulativeEnergy)*norm;
    m_STD = sqrt(m_variance);

    m_DerivativePsiE *= norm;
    m_DeltaPsi *= norm;
    m_EnergyDer = 2*(m_DerivativePsiE - m_DeltaPsi*m_energy);

    m_totalTime = total_time;
    m_acceptedStep = acceptedStep;
}

void Sampler::computeOneBodyDensity(){
  int N = m_system->getNumberOfParticles(); // Number of Particles
  int Dim = m_system->getNumberOfDimensions(); // The Dimension
  double r2;
  double tol = 0.5*(m_system->getBinEndpoint()-m_system->getBinStartpoint())/(m_system->getNumberofBins());
  double step = (m_system->getBinEndpoint() - m_system->getBinStartpoint())/m_system->getNumberofBins();
  for (int i = 0; i < N; i++){
    r2 = 0;
    for (int j = 0; j < Dim; j++){
      r2 += m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
    }
    r2 = sqrt(r2);

    m_system->setParticlesPerBin(int(r2/step) + 1);
  }
}



void Sampler::Analysis(int MCcycles){

  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles
  double Energy = m_cumulativeEnergy * norm;
  m_Energies[MCcycles-1] = Energy;
}


void Sampler::WriteResultstoFile(ofstream& ofile, int MCcycles){
  int N = m_system->getNumberOfParticles(); // Number of Particles
  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles

  double Energy = m_cumulativeEnergy * norm;
  double CumulativeEnergy2 = m_cumulativeEnergy2 *norm;
  double CumulativeEnergy = m_cumulativeEnergy *norm;
  double Variance = CumulativeEnergy2 - CumulativeEnergy*CumulativeEnergy;
  double STD = sqrt(Variance*norm);


  //ofile << "\n";
  //ofile << setw(15) << setprecision(8) << MCcycles; // # Monte Carlo cycles (sweeps per lattice)
  ofile << setw(15) << setprecision(8) << Energy << endl; // Mean energy
  //ofile << setw(15) << setprecision(8) << m_cumulativeEnergy << endl; // Variance
  //ofile << setw(15) << setprecision(8) << STD; // # Standard deviation

}

void Sampler::WriteOneBodyDensitytoFile(ofstream& ofile){
  int N = m_system->getNumberOfParticles();
  int MCcycles = m_system->getNumberOfMetropolisSteps();
  double step = (m_system->getBinEndpoint() - m_system->getBinStartpoint())/m_system->getNumberofBins();
  step = step*step*step;
  double PI = 4*atan(1);
  for (int i = 0; i < m_system->getNumberofBins(); i++){
    double Volume = (4*(i*(i+1) + 1/3)*PI*step);
    ofile << setw(15) << setprecision(8) <<  m_system->getBinVector()[i]; // Mean energy
    ofile << setw(15) << setprecision(8) << m_system->getParticlesPerBin()[i] << endl; // Variance
  }
}
