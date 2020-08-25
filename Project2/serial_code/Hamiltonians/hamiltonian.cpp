#include "hamiltonian.h"
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"
#include "../NeuralNetworks/network.h"


Hamiltonian::Hamiltonian(System* system) {
    m_system = system;
}
