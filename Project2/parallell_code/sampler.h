#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <armadillo>

using namespace arma;
using namespace std;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void setMCcyles(int effectiveSamplings);
    void setacceptedStep(int counter);
    void setEnergies(int OptCycles);
    void setBlocking(int MCcycles);
    void setGradients();
    void sample(int step);
    void printOutputToTerminal(int numberOfProcesses);
    void computeAverages(double time, int numberOfProcesses);
    void Blocking(int MCcycles);
    void Energies(int OptCycle, int totOptCycles);
    void WriteBlockingtoFile(ofstream& ofile);

    double getSTD()                   { return m_globalSTD; }
    double getVAR()                   { return m_globalvariance; }
    double getEnergy()                { return m_energy; }
    double getEnergyDer()             { return m_EnergyDer; }
    double getTime()                  { return m_globalTime; }
    double getAcceptedStep()          { return m_globalacceptedSteps; }
    vec getEnergies()                 { return m_Energies; }
    vec getBlocking()                 { return m_Blocking; }




private:
  int     m_numberOfMetropolisSteps = 0;
  int     m_MCcycles = 0;
  int     m_stepNumber = 0;
  double  m_localenergy = 0;
  double  m_globalenergy = 0;
  double  m_localcumulativeEnergy = 0;
  double  m_globalcumulativeEnergy = 0;
  double  m_localcumulativeEnergy2 = 0;
  double  m_globalcumulativeEnergy2 = 0;

  double  m_DeltaPsi  = 0;
  double  m_DerivativePsiE  = 0;
  double  m_EnergyDer  = 0;
  double  m_energy = 0;

  double  m_localvariance = 0;
  double  m_globalvariance = 0;
  double  m_localSTD = 0;
  double  m_globalSTD = 0;
  double  m_localTime = 0;
  double  m_globalTime = 0;
  double  m_localacceptedSteps = 0;
  double  m_globalacceptedSteps = 0;

  vec m_Energies;
  vec m_Blocking;
  vec m_Times;

  vec m_localaDelta;
  vec m_localbDelta;
  vec m_localwDelta;
  vec m_localEaDelta;
  vec m_localEbDelta;
  vec m_localEwDelta;

  vec m_globalaDelta;
  vec m_globalbDelta;
  vec m_globalwDelta;
  vec m_globalEaDelta;
  vec m_globalEbDelta;
  vec m_globalEwDelta;


    vec m_agrad;
    vec m_bgrad;
    vec m_wgrad;



    class System* m_system = nullptr;
};
