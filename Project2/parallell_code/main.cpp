#include "main.h"

int main(int argc, char **argv) {

  MPI_Init (&argc, &argv);
  // Optimizer parameters
  double eta              = pow(2, -2);   // Learning rate
  int numberOfHidden      = 2;            // Number of hidden units

  // Initialisation parameters
  int numberOfParticles   = 2;
  int numberOfDimensions  = 2;
  int numberOfInputs      = numberOfParticles*numberOfDimensions;  // Number of visible units
  double sigma            = 1.0;          // Normal distribution visibles
  double gibbs            = 1.0;          // Gibbs parameter to change the wavefunction
  bool gaussianInitialization = false;    // Weights & biases (a,b,w) initialized uniformly or gaussian

  // Sampler parameters
  int OptCycles           = 20;          // Number of optimization iterations
  int MCcycles            = pow(2, 22);   // Number of samples in each iteration
  double stepLength       = 1.0;         // Metropolis step length.
  double timeStep         = 0.5;         // Timestep to be used in Metropolis-Hastings
  double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used for equilibration.

  // Hamiltonian parameters
  double omega            = 1.0;          // Oscillator frequency.
  bool includeInteraction = true;        // Include interaction or not

  //Initialise the system.
  System* system = new System();
  system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
  system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
  system->setHamiltonian              (new HarmonicOscillator(system, omega));
  system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
  system->setStepLength               (stepLength);
  system->setEquilibrationFraction    (equilibration);

  system->setTimeStep		      (timeStep);
  system->setImportanceSampling       (true);

  system->setRepulsivePotential       (includeInteraction);
  system->setPrintOutToTerminal       (true);
  system->runOptimizer                (OptCycles, MCcycles);


  MPI_Finalize();
  return 0;
}
