cmake_minimum_required(VERSION 3.20)

project(GPU_Test VERSION 1.0.0 LANGUAGES CXX)

# GPU acceleration support
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration for hash calculations" ON)

if(ENABLE_GPU_ACCELERATION)
    message(STATUS "Checking for GPU acceleration libraries...")

    # Try CUDA first (preferred for NVIDIA GPUs)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
        set(GPU_BACKEND "CUDA")
        set(USE_CUDA ON)
        set(HAS_CUDA ON)
    else()
        message(STATUS "CUDA not found, trying OpenCL...")

        # Fallback to OpenCL (works with AMD/Intel GPUs and MinGW)
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL found: Enabling OpenCL acceleration")
            set(GPU_BACKEND "OpenCL")
            set(USE_OPENCL ON)
            set(HAS_OPENCL ON)
        else()
            message(WARNING "No GPU acceleration libraries found. Building CPU-only version.")
            message(WARNING "To enable GPU acceleration:")
            message(WARNING "  - For NVIDIA: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            message(WARNING "    Then use: cmake .. -G \"Visual Studio 16 2019\" (or appropriate VS version)")
            message(WARNING "  - For AMD/Intel: Install OpenCL development packages")
            set(ENABLE_GPU_ACCELERATION OFF)
        endif()
    endif()

    if(ENABLE_GPU_ACCELERATION)
        message(STATUS "GPU acceleration enabled with backend: ${GPU_BACKEND}")
    endif()
endif()

# GPU-specific compilation
if(ENABLE_GPU_ACCELERATION)
    if(USE_CUDA)
        # CUDA requires MSVC on Windows - check if we're using the right compiler
        if(WIN32 AND NOT MSVC)
            message(FATAL_ERROR "CUDA acceleration requires MSVC compiler on Windows.\n"
                              "Current compiler: ${CMAKE_CXX_COMPILER_ID}\n"
                              "To enable CUDA acceleration:\n"
                              "1. Open 'Visual Studio 2022 Developer Command Prompt' or 'x86 Native Tools Command Prompt for VS 2022'\n"
                              "2. Run: cmake .. -G \"Visual Studio 17 2022\" (or appropriate VS version)\n"
                              "3. Build with: cmake --build . --config Release\n"
                              "Alternatively, you can disable GPU acceleration: cmake .. -DENABLE_GPU_ACCELERATION=OFF")
        else()
            message(STATUS "CUDA compilation enabled with MSVC")
        endif()
    elseif(USE_OPENCL)
        message(STATUS "OpenCL compilation enabled")
    endif()
endif()

message(STATUS "Build configuration complete")
message(STATUS "GPU Backend: ${GPU_BACKEND}")
message(STATUS "MSVC Compiler: ${MSVC}")
message(STATUS "Compiler ID: ${CMAKE_CXX_COMPILER_ID}")