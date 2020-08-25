cmake_minimum_required(VERSION 3.16)
project(project_2_fys4411)

# Set compiler
set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)

find_package(MPI REQUIRED)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    WaveFunctions
    NeuralNetworks
    Block
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
    GLOB SOURCES "NeuralNetworks/*.cpp"
    GLOB SOURCES "Block/*.cpp"
)

if(APPLE)
  # If problems, add these two lines (I think)
  # set(CMAKE_CXX_COMPILER "/usr/local/Cellar/open-mpi/4.0.3/lib/libmpi.dylib")
  # set(CMAKE_C_COMPILER "/usr/local/Cellar/open-mpi/4.0.3/lib/libmpi.dylib")
  set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")

else()
  set (CMAKE_CXX_FLAGS "-Wall -O2 -fopenmp -lopenblas")
endif()

# Including armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(PROJECT_EXECUTABLE vmc)

# Including MPI
include_directories(${MPI_INCLUDE_PATH})


# Create executable called "vmc"
add_executable(${PROJECT_EXECUTABLE} ${SOURCES})


# Linking armadillo and MPI
target_link_libraries(${PROJECT_EXECUTABLE} ${ARMADILLO_LIBRARIES})
target_link_libraries(${PROJECT_EXECUTABLE} ${MPI_LIBRARIES})


# setting MPI with correct compilation flags
if(MPI_COMPILE_FLAGS)
  set_target_properties(${PROJECT_EXECUTABLE} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()


# Linking the target with MPI
if(MPI_LINK_FLAGS)
  set_target_properties(${PROJECT_EXECUTABLE} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
