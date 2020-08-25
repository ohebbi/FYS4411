#include <cassert>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#include "system.h"
#include "sampler.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "NeuralNetworks/network.h"


bool System::metropolisStep() {
  /* Perform the actual Metropolis step: Choose a particle at random and
   * change it's position by a random amount, and check if the step is
   * accepted by the Metropolis test (compare the wave function evaluated
   * at this new position with the one at the old position).
   */

   // Set up the uniform distribution for x \in [[0.0, 1.0]
   std::uniform_real_distribution<double> Uniform(0.0,1.0);
   // Set up the uniform distribution for x \in [[-0.5, 0.5]
   std::uniform_real_distribution<double> RandomNumberGenerator(-0.5,0.5);

   // Set up the uniform distribution for x \in [[0, N]
   std::uniform_int_distribution<int> Inputs(0, getNumberOfInputs()-1);

   int updateCoordinate = Inputs(m_randomEngine);
   double a = RandomNumberGenerator(m_randomEngine); // Random number

   double wfold, wfnew;

   // Initial Position
   wfold = getWaveFunction()->evaluate();
   vec posold = getNetwork()->getPositions();

   vec posnew = getNetwork()->getPositions();

   // Trial position moving one particle at the time in all dimensions
   posnew(updateCoordinate) += m_stepLength*a;
   getNetwork()->setPositions(posnew);

   wfnew = getWaveFunction()->evaluate();

   double probRatio = (wfnew*wfnew)/(wfold*wfold);

   // Metropolis test
   if ( (1.0 < probRatio) || (Uniform(m_randomEngine) < probRatio) ){
      return true;
   }

   // return to previous value if Metropolis test is false
   else{
      getNetwork()->setPositions(posold);
      return false;
    }
}


bool System::ImportanceMetropolisStep() {
     // Perform the Importance sampling metropolis step.

     // Set up the uniform distribution for x \in [[0, 1]
     std::normal_distribution<double> Normal(0.0,1.0);
     std::uniform_real_distribution<double> Uniform(0.0,1.0);
     // Set up the uniform distribution for x \in [[0, N]
     std::uniform_int_distribution<int> Inputs(0,getNumberOfInputs()-1);

     int updateCoordinate = Inputs(m_randomEngine);
     double a = Normal(m_randomEngine); // Random number

     double wfold, wfnew, poschange;

     // Initial Position
     vec posold = getNetwork()->getPositions();
     wfold = getWaveFunction()->evaluate();
     vec qfold = 2*(getWaveFunction()->computeFirstDerivative());

     vec posnew = getNetwork()->getPositions();

     // Trial position moving one particle at the time in all dimensions
     poschange = a*sqrt(m_timeStep) + qfold(updateCoordinate)*m_timeStep*m_diffusionCoefficient;
     posnew(updateCoordinate) += poschange;
     getNetwork()->setPositions(posnew);

     // evaluate new position
     wfnew = getWaveFunction()->evaluate();
     vec qfnew = 2*(getWaveFunction()->computeFirstDerivative());

     // Greens function
     double greensFunction = 0;
     greensFunction += 0.5*(qfold(updateCoordinate) + qfnew(updateCoordinate))*(m_diffusionCoefficient*m_timeStep*0.5*(qfold(updateCoordinate) - qfnew(updateCoordinate)) - posnew(updateCoordinate) + posold(updateCoordinate));
     greensFunction = exp(greensFunction);

     double probRatio = (greensFunction*wfnew*wfnew)/(wfold*wfold);

     // #Metropolis-Hastings test to see whether we accept the move
    if ( (1.0 < probRatio) || (Uniform(m_randomEngine) < probRatio) ){

    return true;
  }

  // return to previous value if Metropolis test is false
  else{

    getNetwork()->setPositions(posold);
    return false;
  }

}

void System::Gibbs() {
  // Set new hidden variables given positions, according to the logistic sigmoid function
  // (implemented by comparing the sigmoid probability to a uniform random variable)

  // Set up the uniform distribution for x \in [[0, 1]
  std::normal_distribution<double> Normal(0.0,1.0);
  std::uniform_real_distribution<double> Uniform(0.0,1.0);
  // Set up the uniform distribution for x \in [[0, N]
  std::uniform_int_distribution<int> Inputs(0,getNumberOfInputs()-1);

  int nx = getNumberOfInputs();
  int nh = getNumberOfHidden();

  double sigma = getWaveFunction()->getParameters()[0];
  double sigma2 = sigma*sigma;

  vec x = getNetwork()->getPositions();
  vec a = getNetwork()->getBiasA();
  vec b = getNetwork()->getBiasB();
  mat w = getNetwork()->getWeigths();

  vec h(nh);
  vec posnew(nx);

  double z, logisticSigmoid;
  for (int j = 0; j < nh; j++) {
    z = b(j) + (dot(x, w.col(j)))/(sigma2);
    logisticSigmoid = 1.0/(1+exp(-z));
    h(j) = Uniform(m_randomEngine) < logisticSigmoid;
  }

  // Set new positions (visibles) given hidden, according to normal distribution
  std::normal_distribution<double> distributionX;
  double xMean;
  for (int i = 0; i < nx; i++) {
      xMean = a(i) + dot(w.row(i), h);
      distributionX = std::normal_distribution<double>(xMean, sigma);
      posnew(i) = distributionX(m_randomEngine);
  }

  getNetwork()->setPositions(posnew);
}




void System::runOptimizer(int OptCycles, int numberOfMetropolisSteps) {

  MPI_Comm_rank (MPI_COMM_WORLD, &m_myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &m_numberOfProcesses);

  m_sampler                   = new Sampler(this);
  m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
  m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);

  // Initialize the seed and call the Mersienne algo
  std::random_device rd;
  m_randomEngine = std::mt19937_64(rd());

  double start_time, end_time, localTime;

  m_sampler->setEnergies(OptCycles);
  m_sampler->setBlocking(numberOfMetropolisSteps);

  for (int i = 0; i < OptCycles; i++){
    start_time = omp_get_wtime();

    runMetropolisSteps(numberOfMetropolisSteps);

    end_time = omp_get_wtime();
    localTime = end_time - start_time;

    m_sampler->computeAverages(localTime, m_numberOfProcesses);
    if (m_myRank==0){
      m_sampler->Energies(i, OptCycles);
      m_sampler->printOutputToTerminal(m_numberOfProcesses);
    }

  }
  
  //if (m_myRank==0){
  //  m_sampler->WriteBlockingtoFile(ofile);
  //}
  
}


void System::runMetropolisSteps(int numberOfMetropolisSteps) {

  double counter = 0;
  bool acceptedStep;
  double effectivesampling = 0;
  int eq = getEquilibrationFraction()*numberOfMetropolisSteps;


  for (int i = 1; i <= numberOfMetropolisSteps + eq; i++) {

    // Choose importance samling, gibbs sampling or brute force
    if (getImportanceSampling()){
        acceptedStep = ImportanceMetropolisStep();
    }

    else if (getGibbsSampling()){
      Gibbs();
    }

    else{
        acceptedStep = metropolisStep();
    }

    if (i > getEquilibrationFraction() * numberOfMetropolisSteps){
      effectivesampling++;
      counter += acceptedStep;

      m_sampler->sample(effectivesampling);
      if (m_myRank==0){
          m_sampler->Blocking(effectivesampling);
      }
    }
  }
  m_sampler->setMCcyles(effectivesampling);
  m_sampler->setacceptedStep(counter);

}

// A lot of setters.
void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
}

void System::setNumberOfHidden(int numberOfHidden) {
    m_numberOfHidden = numberOfHidden;
}

void System::setStepLength(double stepLength) {
    assert(stepLength >= 0);
    m_stepLength = stepLength;
}

void System::setTimeStep(double timeStep) {
    assert(timeStep >= 0);
    m_timeStep = timeStep;
}

void System::setStepSize(double stepSize) {
    assert(stepSize >= 0);
    m_stepSize = stepSize;
}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}

void System::setHamiltonian(Hamiltonian* hamiltonian) {
    m_hamiltonian = hamiltonian;
}
void System::setNetwork(Network* network) {
    m_network = network;
}

void System::setWaveFunction(WaveFunction* waveFunction) {
    m_waveFunction = waveFunction;
}

void System::setBinVector(double binStartpoint, double binEndpoint, int numberofBins){
  std::vector<double> binVector;
  std::vector<int> binCounter;

  double step = (binEndpoint-binStartpoint)/(numberofBins);
  for (int i = 0; i < numberofBins; i++){
    binVector.push_back((double)i * step);
    binCounter.push_back(0);
  }
  m_binVector = binVector;
  m_binCounter = binCounter;
  m_partclesPerBin = binCounter;
}

void System::setBinCounter(int new_count, int index){
  m_binCounter[index] = new_count;
}

void System::setParticlesPerBin(int index){
  m_binCounter[index]++;
}

void System::setOneBodyDensity(bool oneBodyDensity){
  m_oneBodyDensity = oneBodyDensity;
}

void System::setInitialState(InitialState* initialState) {
    m_initialState = initialState;
}


void System::setDiffusionCoefficient(double diffusionCoefficient) {
    m_diffusionCoefficient = diffusionCoefficient;
}

void System::setBinEndpoint(double binEndpoint) {
    m_binEndpoint = binEndpoint;
}

void System::setBinStartpoint(double binStartpoint) {
    m_binStartpoint = binStartpoint;
}

void System::setNumberofBins(int numberofBins) {
    m_numberofBins = numberofBins;
}



bool System::setRepulsivePotential(bool statement){
  m_statement = statement;
  return m_statement;
}

bool System::setImportanceSampling(bool importance_sampling){
  m_importance_sampling = importance_sampling;
  return m_importance_sampling;
}

bool System::setGibbsSampling(bool gibbs_sampling){
  m_gibbs_sampling = gibbs_sampling;
  return m_gibbs_sampling;
}

bool System::setOptimizer(bool optimizer){
  m_optimizer = optimizer;
  return m_optimizer;
}

bool System::setPrintOutToTerminal(bool print_terminal){
  m_print_terminal = print_terminal;
  return m_print_terminal;
}
