# Simple Variational Monte Carlo solve for FYS4411
This class structure is forked and inspired by mortele (spring 2016) (https://github.com/mortele/variational-monte-carlo-fys4411), and is the main repository.

## Code skeleton
This is to give an overview of the repository without going in-depth of the details, as this project is quite extensive (but that is of course debatable).

  - Hamiltonians
    - harmonicoscillator.cpp (where energies are calculated)
    - ... classes of hamiltonian
  - InitialStates
    - randomuniform.cpp (where initial state is given)
    - ... classes of initialstate
  - Math (directory with classes of random and initialisers)
  - WaveFunctions
    - simplegaussian.cpp (where one evaluates the wavefunction in different forms)
    - ... classes of wavefunction
  - Python
    - Variational_Monte_Carlo.ipynb (all results used in report is stated here for easy reproducibility)
    - Results (with a lot of .PGFs and .dat files)
      - task_a
      - (to)
      - task_g
      - statistical_tools
  - report
    - report.pdf (the report of the project.)
  - CMakeLists.txt (set flags and add executable)
  - compile_project (see how to compile project below)
  - main.cpp (main program, more details below)
  - Makefile
  - particle.cpp (class of particle)
  - particle.h (... and its header file)
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

#### Cleaning the directory
Run `make clean` in the top-directory to remove the executable `vmc` and the `build`-directory.

#### Windows
Compilation of the project using Windows is still an open question to me, but please include a pull-request if you've got an example. CMake should be OS-independent, but `make` does not work on Windows.

## Running the program (code in main.cpp)

Runnning the program will firstly make you choose which task you would like to choose, starting from b to f.

Then it will ask how many Monte Carlo cycles. Pay attention to what it asks you about.

After that, insert the number of particles and dimensions.

And voilá, the results will be found in the Python/Results folder under their respective task, and the Jupyter Notebook will then automatically renew all the figures. This will however not update the report, and we will leave the latest updated Notebook updated with the report added. 

## Last updated 24.03.2020
