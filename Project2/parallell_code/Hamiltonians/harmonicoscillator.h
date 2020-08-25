#pragma once
#include "hamiltonian.h"
#include <armadillo>
#include <vector>

class HarmonicOscillator : public Hamiltonian {
public:
    HarmonicOscillator(System* system, double omega);
    double computeLocalEnergy(Network* network);
    double Interaction();


private:
    double m_omega = 0;
};
