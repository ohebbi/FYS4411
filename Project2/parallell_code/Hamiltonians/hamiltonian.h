#pragma once
#include <vector>
#include <armadillo>
#include "../NeuralNetworks/network.h"

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy(Network* network) = 0;
    virtual double Interaction() = 0;


protected:
    class System* m_system = nullptr;

};
