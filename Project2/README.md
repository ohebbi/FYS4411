# Project 2 - Restricted Boltzmann Machine
This class structure is forked and inspired by mortele (spring 2016) (https://github.com/mortele/variational-monte-carlo-fys4411), and is the main repository.

## serial_code skeleton
This is to give an overview of the repository inside serial_code without going in-depth of the details. Same structure has also been used for parallell_code, however with some adjustment for running on the supercomputer Saga. 

  - Hamiltonians
    - harmonicoscillator.cpp (where energies are calculated)
    - ... classes of hamiltonian
  - InitialStates
    - randomuniform.cpp (where initial state is given)
    - ... classes of initialstate
  - Neural Networks
    - neuralnetwork.cpp (where one computes gradients of the network and utilize stochastic gradient descent)
  - WaveFunctions
    - neuralquantumstate.cpp (where one evaluates the wavefunction in different forms)
    - ... classes of wavefunction
  - Python
    - Restricted_Boltzmann_Machine.ipynb (all results used in report is stated here for easy reproducibility)
    - Results (with a lot of .PGFs and .dat files)
      - Brute_Force
      - Importance_Sampling
      - Gibbs
      - Statistical_Analysis
    - statistical_tools (python programs for statistical analysis)
  - CMakeLists.txt (set flags and add executable)
  - compile_project (see how to compile project below)
  - main.cpp (main program, more details below)
  - Makefile
  - README.md (this. du-uh)
  - sampler.cpp (where all and everything is sampled)
  - sampler.h (aand its headerfile)
  - system.cpp (where everything important about MC and Metropolis is)
  - system.h (...)
  - variational-monte-carlo-fys4411.pro (needed for compilation for QT-creator)


## Dependencies  
Please make sure that the following packages are installed on your computer.
  - iostream
  - iomanip
  - fstream
  - random
  - string
  - cmath
  - armadillo
  - cassert
  - math.h
  - omp.h
  - time.h

## Compilling and running the project
There are now several options you can use for compiling the project. If you use QT Creator, you can import this project into the IDE and point it to the `.pro`-file. If not, you can use CMake to create a Makefile for you which you can then run. You can install CMake through one of the Linux package managers, e.g., `apt install cmake`, `pacman -S cmake`, etc. For Mac you can install using `brew install cmake`. Other ways of installing are shown here: [https://cmake.org/install/](https://cmake.org/install/).

### Compiling the project using CMake
In a Linux/Mac terminal this can be done by the following commands
```bash
# Create build-directory
mkdir build

# Move into the build-directory
cd build

# Run CMake to create a Makefile
cmake ../

# Make the Makefile using two threads
make -j2

# Move the executable to the top-directory
mv vmc ..
```
Or, simply run the script `compile_project` via
```bash
./compile_project
```
and the same set of commands are done for you. Now the project can be run by executing
```bash
./vmc
```
in the top-directory. This is how the authors of this project prefer to run it.

If one meets some problems with compiling problems for gcc running macOS Catalina 10.15.4, add this line to CMakeLists.txt
```bash
cmake -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 ../
```

#### Cleaning the directory
Run `make clean` in the top-directory to remove the executable `vmc` and the `build`-directory.

#### Windows
Compilation of the project using Windows is still an open question to me, but please include a pull-request if you've got an example. CMake should be OS-independent, but `make` does not work on Windows.

## Running the program (code in main.cpp)

Runnning the program will only make you choose which sampler algorithm you would like to choose, starting from 1 to 3.

And voilá, the results will be found in the Python/Results folder under their respective task, and the Jupyter Notebook will then automatically renew all the figures. This will however not update the report, and we will leave the latest updated Notebook updated with the report added.

## Last updated 27.05.2020
