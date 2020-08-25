#pragma once
#include "hamiltonian.h"
#include <vector>

class HarmonicOscillator : public Hamiltonian {
public:
    HarmonicOscillator(System* system, double omega);
    double computePotentialEnergy(std::vector<Particle*> particles);
    double computeRepulsiveInteraction(std::vector<Particle*> particles);
    double computeLocalEnergyAnalytical(std::vector<Particle*> particles);
    double computeLocalEnergyNumerical(std::vector<Particle*> particles);
    std::vector<double> computeQuantumForce(std::vector<Particle*> particles, int i);

private:
    double m_omega = 0;
};
