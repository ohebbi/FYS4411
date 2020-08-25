#pragma once
#include "initialstate.h"
#include <random>
#include <iostream>

class RandomUniform : public InitialState {
public:
    RandomUniform(System* system, int numberOfDimensions, int numberOfParticles, int numberOfHidden, bool gaussianInitialization);
    void setupInitialState(bool gaussianInitialization);

private:
  std::mt19937_64 m_randomEngine; // For the distributions
  int m_numberOfDimensions = 0;
  int m_numberOfParticles = 0;
  int m_numberOfInputs = 0;
  int m_numberOfHidden = 0;

};
