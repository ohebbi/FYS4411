#include "randomuniform.h"
#include <iostream>
#include <cassert>
#include <random>
#include "Math/random.h"
#include "../particle.h"
#include "../system.h"

using namespace  std;
using std::cout;
using std::endl;

RandomUniform::RandomUniform(System*    system,
                             int        numberOfDimensions,
                             int        numberOfParticles)  :
        InitialState(system) {
    assert(numberOfDimensions > 0 && numberOfParticles > 0);
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParticles  = numberOfParticles;

    /* The Initial State class is in charge of everything to do with the
     * initialization of the system; this includes determining the number of
     * particles and the number of dimensions used. To make sure everything
     * works as intended, this information is passed to the system here.
     */
    m_system->setNumberOfDimensions(numberOfDimensions);
    m_system->setNumberOfParticles(numberOfParticles);
    setupInitialState();
}

void RandomUniform::setupInitialState() {

  // Initialize the seed and call the Mersienne algo
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // Set up the uniform distribution for x \in [[0, 1]
  std::uniform_real_distribution<double> Uniform(0.0,1.0);
  std::normal_distribution<double> Normal(0.0,1.0);

    for (int i=0; i < m_numberOfParticles; i++) {
        std::vector<double> position = std::vector<double>();

        for (int j=0; j < m_numberOfDimensions; j++) {
            if (m_system->getImportanceSampling()){
                  position.push_back(Normal(gen) *sqrt(m_system->getTimeStep()) );
            }
            else{
              position.push_back(Uniform(gen) - 0.5 );
            }
        }
        m_particles.push_back(new Particle());
        m_particles.at(i)->setNumberOfDimensions(m_numberOfDimensions);
        m_particles.at(i)->setPosition(position);
    }
}
