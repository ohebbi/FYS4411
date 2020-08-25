#pragma once
#include <vector>

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computePotentialEnergy(std::vector<class Particle*> particles) = 0;
    virtual double computeRepulsiveInteraction(std::vector<class Particle*> particles) = 0;
    virtual double computeLocalEnergyAnalytical(std::vector<class Particle*> particles) = 0;
    virtual double computeLocalEnergyNumerical(std::vector<class Particle*> particles) = 0;
    virtual std::vector<double> computeQuantumForce(std::vector<class Particle*> particles, int i) = 0;

protected:
    class System* m_system = nullptr;
};
