project(poisson)

cmake_minimum_required(VERSION 2.6)

enable_language(C)

# CMake Options to enable/disable MPI and OpenMP
option(ENABLE_OPENMP "Enable OpenMP support?" ON)
option(ENABLE_MPI    "Enable MPI support?" ON)

if(ENABLE_MPI)
  find_package(MPI)
endif()

if(MPI_FOUND)
  add_definitions(-DHAVE_MPI=1)
  set(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})
  set(DEPLIBS_C ${DEPLIBS_C} ${MPI_C_LIBRARIES})
endif()

if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    add_definitions(-DHAVE_OPENMP=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  endif()
endif()

# Use C99
set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")

# Send the include directories to the compiler
include_directories(${INCLUDES})

# The C based poisson application
add_executable(poisson poisson.c)

# Link to libraries
target_link_libraries(poisson m ${MPI_C_LIBRARIES})
