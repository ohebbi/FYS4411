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
#include <mpi.h>

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

using namespace arma;
using namespace std;

// #############################################################################
// ############################### Description: ################################
// #############################################################################
// This is the main program main.cpp where one chooses the constants, initialise the
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
