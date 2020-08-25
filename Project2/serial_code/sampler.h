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
    void printOutputToTerminal(double time);
    void computeAverages(double time);
    void Blocking(int MCcycles);
    void Energies(int OptCycles);
    void WriteBlockingtoFile(ofstream& ofile);

    double getSTD()                   { return m_STD; }
    double getVAR()                   { return m_variance; }
    double getEnergy()                { return m_energy; }
    double getEnergyDer()             { return m_EnergyDer; }
    double getTime()                  { return m_totalTime; }
    double getAcceptedStep()          { return m_acceptedStep; }
    vec getEnergies()                 { return m_Energies; }
    vec getBlocking()                 { return m_Blocking; }




private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_MCcycles = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    double  m_DeltaPsi  = 0;
    double  m_DerivativePsiE  = 0;
    double  m_EnergyDer  = 0;
    double  m_variance = 0;
    double  m_STD = 0;
    double  m_totalTime = 0;
    double  m_acceptedStep = 0;

    vec m_Energies;
    vec m_Blocking;


    vec m_aDelta;
    vec m_bDelta;
    vec m_wDelta;


    vec m_EaDelta;
    vec m_EbDelta;
    vec m_EwDelta;


    vec m_agrad;
    vec m_bgrad;
    vec m_wgrad;



    class System* m_system = nullptr;
};
