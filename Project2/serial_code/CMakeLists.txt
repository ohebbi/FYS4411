cmake_minimum_required(VERSION 3.17)
project(project_2_fys4411)

# Add directories containing header-files
include_directories(
    .
    Hamiltonians
    InitialStates
    WaveFunctions
    NeuralNetworks
)

# Add sources from all directories
file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
    GLOB SOURCES "NeuralNetworks/*.cpp"

)

if(APPLE)
  # If problems with flags, export these in unix
  #export LDFLAGS="-L/usr/local/opt/llvm/lib"
  #export CPPFLAGS="-I/usr/local/opt/llvm/include"

  set (CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
else()
  set (CMAKE_CXX_FLAGS "-fopenmp")
endif()

# Including armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


# Including MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


# Create executable called "vmc"
add_executable(vmc ${SOURCES})

# Linking armadillo and MPI
target_link_libraries(vmc ${ARMADILLO_LIBRARIES})
target_link_libraries(vmc ${MPI_LIBRARIES})

# setting MPI with correct compilation flags
if(MPI_COMPILE_FLAGS)
  set_target_properties(vmc PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# Linking the target with MPI
if(MPI_LINK_FLAGS)
  set_target_properties(vmc PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()
