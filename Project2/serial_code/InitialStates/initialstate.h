#pragma once
#include <vector>
#include <random>
#include <iostream>

class InitialState {
public:
    InitialState(class System* system);
    virtual void setupInitialState(bool gaussianInitialization) = 0;

protected:
    class System* m_system = nullptr;
    std::mt19937_64 m_randomEngine; // For the distributions
    int m_numberOfDimensions = 0;
    int m_numberOfParticles = 0;
    int m_numberOfInputs = 0;
    int m_numberOfHidden = 0;

};
