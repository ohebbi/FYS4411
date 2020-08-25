// Dependencies
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <math.h>
#include <cassert>
#include <time.h>
#include <armadillo>
#include <omp.h>

// Include all header files
#include "system.h"
#include "sampler.h"
#include "NeuralNetworks/network.h"
#include "NeuralNetworks/neuralnetwork.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/neuralquantumstate.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"

ofstream ofile;
using namespace arma;
using namespace std;

// #############################################################################
// ############################### Description: ################################
// #############################################################################
// This is the main program where one chooses the constants, initialise the
// system, and eventually adjust the variational parameters as well.
// This file is heavily dependent on what kind of sampler algorithm you want to execute,
// however, it is quite easy to add a more general program
// for a normal Monte Carlo run with your chosen parameters.
//
// #############################################################################
// ########################## Optional parameters: #############################
// #############################################################################
//    // Optimizer parameters
//    double eta              = pow(2, -2);   // Learning rate
//    int numberOfHidden      = 2;            // Number of hidden units


//    // Initialisation parameters
//    int numberOfParticles   = 2;
//    int numberOfDimensions  = 2;
//    int numberOfInputs      = numberOfParticles*numberOfDimensions;  // Number of visible units
//    float sigma            = 0.75;          // Normal distribution visibles
//    double gibbs            = 1.0;          // Gibbs parameter to change the wavefunction // set gibbs = 2 if setGibbsSampling == true
//    bool gaussianInitialization = false; // Weights & biases (a,b,w) initialized uniformly or gaussian


//    // Sampler parameters
//    int OptCycles           = 500;          // Number of optimization iterations
//    int MCcycles            = pow(2, 20);   // Number of samples in each iteration
//    double stepLength       = 1.0;         // Metropolis step length.
//    double timeStep         = 0.5;         // Timestep to be used in Metropolis-Hastings
//    double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
//    double equilibration    = 0.1;          // Amount of the total steps used for equilibration.

//    // Hamiltonian parameters
//    double omega            = 1.0;          // Oscillator frequency.
//    bool includeInteraction = true;         // Include interaction or not
//
// #############################################################################
// ############# How to initialise a system and its parameters #################
// #############################################################################
//   - Some parameters have to be initialised for a system to run, while others
//     are not needed.
//     The ones that HAS to be initialised is:
//
// #############################################################################
//    System* system = new System();
//    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
//    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
//    system->setHamiltonian              (new HarmonicOscillator(system, omega));
//    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
//    system->setStepLength               (stepLength);
//    system->setEquilibrationFraction    (equilibration);
//    system->setPrintOutToTerminal       (true); // true or false
//    system->runOptimizer                (ofile, OptCycles, MCcycles);
// #############################################################################
//  - The optional initialiser:
// #############################################################################
//    system->setRepulsivePotential       (true); // true or false
//    system->setImportanceSampling       (true);  // true or false
//    system->setGibbsSampling            (true); // true or false
//    system->setTimeStep                 (timeStep); // set if setImportanceSampling == true
//    system->setDiffusionCoefficient     (diffusionCoefficient); // set if setImportanceSampling == true
// #############################################################################


int main() {

  cout << "\n" << "Which sampling algorithm do you want to run?: " << endl;
  cout << "\n" << "Brute force sampling: " <<  "Write 1 " << endl;
  cout << "\n" << "Importance sampling: " <<  "Write 2 " << endl;
  cout << "\n" << "Gibbs sampling: " <<  "Write 3 " << endl;



  // Chosen parameters

  // Optimizer parameters
  double eta              = pow(2, -2);   // Learning rate
  int numberOfHidden      = 2;            // Number of hidden units


  // Initialisation parameters
  int numberOfParticles   = 1;
  int numberOfDimensions  = 1;
  int numberOfInputs      = numberOfParticles*numberOfDimensions;  // Number of visible units
  double sigma            = 1.0;          // Normal distribution visibles
  double gibbs            = 1.0;          // Gibbs parameter to change the wavefunction
  bool gaussianInitialization = false;    // Weights & biases (a,b,w) initialized uniformly or gaussian


  // Sampler parameters
  int OptCycles           = 500;          // Number of optimization iterations
  int MCcycles            = pow(2, 17);   // Number of samples in each iteration
  double stepLength       = 1.0;         // Metropolis step length.
  double timeStep         = 0.5;         // Timestep to be used in Metropolis-Hastings
  double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used for equilibration.

  // Hamiltonian parameters
  double omega            = 1.0;          // Oscillator frequency.
  bool includeInteraction = false;        // Include interaction or not


  cout << "\n" << "Write here " << endl;
  int sampler;
  cin >> sampler;

  // Parameter for writing to files
  string file;
  int gamma = log2(eta);
  int mc = log2(MCcycles);


  if (sampler == 1){

    cout << "-------------- \n" << "Brute force sampling \n" << "-------------- \n" << endl;

    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Statistical_Analysis/BF_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Statistical_Analysis/I_BF_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setStepLength               (stepLength);
    system->setEquilibrationFraction    (equilibration);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();


    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Brute_Force/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Brute_Force/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }

  if (sampler == 2){

    cout << "-------------- \n" << "Importance sampling \n" << "-------------- \n" << endl;

    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Statistical_Analysis/IS_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Statistical_Analysis/I_IS_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setTimeStep                 (timeStep);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->setEquilibrationFraction    (equilibration);
    system->setImportanceSampling       (true);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();


    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Importance_Sampling/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Importance_Sampling/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }



  if (sampler == 3){

    cout << "-------------- \n" << "Gibbs sampling \n" << "-------------- \n" << endl;
    gibbs = 2;

    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Statistical_Analysis/GI_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + "sigma_" + to_string(sigma) + ".dat";
    //file = "Python/Results/Statistical_Analysis/I_GI_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + "sigma_" + to_string(sigma) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setGibbsSampling            (true);
    system->setEquilibrationFraction    (equilibration);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();

    // Choose which file to write to, either non-interaction or interaction
    //file = "Python/Results/Gibbs/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + "sigma_" + to_string(sigma) + ".dat";
    //file = "Python/Results/Gibbs/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + "sigma_" + to_string(sigma) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }


  return 0;
}
