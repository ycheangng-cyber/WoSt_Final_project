cmake_minimum_required(VERSION 3.10)
project(WoSt)

# ============================================================================
# C++ Standard
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# TBB
# ============================================================================

set(CMAKE_PREFIX_PATH
    "/opt/intel/oneapi/tbb/2022.3"
    ${CMAKE_PREFIX_PATH}
)

find_package(TBB REQUIRED CONFIG)
find_package(Threads REQUIRED)

# ============================================================================
# OpenMP for AppleClang
# ============================================================================

if(APPLE)

    # Homebrew libomp
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")

    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY
        /opt/homebrew/opt/libomp/lib/libomp.dylib
    )

endif()

find_package(OpenMP REQUIRED)

# ============================================================================
# Include directories
# ============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ============================================================================
# Sources
# ============================================================================

set(SOURCES
    main.cpp
    src/WoStGeometryBackend.cpp
    src/WoStKernel.cpp
)

# ============================================================================
# Executable
# ============================================================================

add_executable(wost ${SOURCES})

# ============================================================================
# Compile options
# ============================================================================

target_compile_options(wost PRIVATE
    -O3
    -Wall
    -Wextra
    -funroll-loops
    -ffast-math
)

# ============================================================================
# Link
# ============================================================================

target_link_libraries(wost
    PRIVATE
    TBB::tbb
    Threads::Threads
    OpenMP::OpenMP_CXX
)