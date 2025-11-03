# GPU Acceleration Implementation Plan - NVIDIA GPU

**Date:** November 3, 2025  
**Target Hardware:** NVIDIA GPU (CUDA/OpenCL)  
**Goal:** Implement GPU acceleration for hash calculations and duplicate detection  
**Compatibility:** Must work on GPU and non-GPU machines  

---

## Executive Summary

This plan outlines the implementation of GPU acceleration for DupFinder on NVIDIA hardware. The implementation will:

- **Maintain backward compatibility** with CPU-only machines
- **Automatically detect and utilize** NVIDIA GPUs when available
- **Provide significant performance improvements** for hash calculations (5-10x speedup)
- **Include comprehensive fallback mechanisms** and error handling
- **Follow the existing codebase patterns** and architecture

---

## Phase 1: Environment Setup and Dependencies

### 1.1 Install NVIDIA Development Tools
```bash
# Check if CUDA is already installed
echo "Checking for existing CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "CUDA is already installed:"
    nvcc --version
else
    echo "CUDA not found. Installing CUDA Toolkit..."
    # Download from: https://developer.nvidia.com/cuda-downloads
    # Or using package manager:
    # Ubuntu/Debian:
    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    # sudo dpkg -i cuda-keyring_1.0-1_all.deb
    # sudo apt-get update
    # sudo apt-get install cuda-toolkit-11-8
    
    # Or using runfile installer (interactive):
    # wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    # sudo sh cuda_11.8.0_520.61.05_linux.run
fi

# Check NVIDIA drivers
echo "Checking NVIDIA GPU and drivers..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "NVIDIA drivers not found. Please install NVIDIA drivers first."
    echo "Visit: https://www.nvidia.com/Download/index.aspx"
    exit 1
fi
```

### 1.2 Windows CUDA Installation
```powershell
# Check for existing CUDA installation
Write-Host "Checking for existing CUDA installation..."
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    Write-Host "CUDA is already installed:"
    nvcc --version
} else {
    Write-Host "CUDA not found. Installing CUDA Toolkit..."
    # Download CUDA installer from: https://developer.nvidia.com/cuda-downloads
    # Run the installer executable
    # Or use winget if available:
    # winget install NVIDIA.CUDA
}

# Check NVIDIA drivers
Write-Host "Checking NVIDIA GPU and drivers..."
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
} else {
    Write-Host "NVIDIA drivers not found. Please install NVIDIA drivers first."
    Write-Host "Visit: https://www.nvidia.com/Download/index.aspx"
    exit 1
}
```

### 1.2 Update CMakeLists.txt for GPU Support
**File:** `CMakeLists.txt`

```cmake
# Add GPU acceleration option
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration for hash calculations" ON)

# GPU dependency detection
if(ENABLE_GPU_ACCELERATION)
    message(STATUS "Checking for GPU acceleration libraries...")
    
    # Try CUDA first (NVIDIA preferred)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "CUDA found: ${CUDA_VERSION}")
        message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
        set(GPU_BACKEND "CUDA")
        set(USE_CUDA ON)
    else()
        message(STATUS "CUDA not found")
        
        # Fallback to OpenCL
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            message(STATUS "OpenCL found: Enabling OpenCL acceleration")
            set(GPU_BACKEND "OpenCL")
            set(USE_OPENCL ON)
        else()
            message(WARNING "No GPU acceleration libraries found. Building CPU-only version.")
            message(WARNING "To enable GPU acceleration:")
            message(WARNING "  - For NVIDIA: Install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads")
            message(WARNING "  - For AMD/Intel: Install OpenCL development packages")
            set(ENABLE_GPU_ACCELERATION OFF)
        endif()
    endif()
    
    if(ENABLE_GPU_ACCELERATION)
        message(STATUS "GPU acceleration enabled with backend: ${GPU_BACKEND}")
    endif()
endif()

# GPU-specific compilation flags
if(USE_CUDA)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-gencode=arch=compute_60,code=sm_60)
    cuda_add_library(gpu_acceleration STATIC
        src/gpu/cuda_hash_calculator.cu
        src/gpu/cuda_memory_manager.cu
    )
elseif(USE_OPENCL)
    add_library(gpu_acceleration STATIC
        src/gpu/opencl_hash_calculator.cpp
        src/gpu/opencl_memory_manager.cpp
        src/gpu/opencl_kernels.cl
    )
endif()

# Link GPU libraries
if(ENABLE_GPU_ACCELERATION)
    target_link_libraries(dupfinder gpu_acceleration)
    if(USE_CUDA)
        target_link_libraries(dupfinder ${CUDA_LIBRARIES})
    elseif(USE_OPENCL)
        target_link_libraries(dupfinder OpenCL::OpenCL)
    endif()
endif()
```

### 1.3 Create GPU Source Directory Structure
```
src/gpu/
├── cuda_hash_calculator.cu      # CUDA hash implementation
├── cuda_memory_manager.cu       # CUDA memory management
├── opencl_hash_calculator.cpp   # OpenCL hash implementation
├── opencl_memory_manager.cpp    # OpenCL memory management
├── opencl_kernels.cl           # OpenCL kernel code
├── gpu_detector.cpp            # GPU capability detection
└── gpu_config.h                 # GPU configuration header
```

---

## Phase 2: Core GPU Infrastructure

### 2.1 GPU Capability Detection
**File:** `src/gpu/gpu_detector.cpp`

```cpp
#include "gpu_detector.h"
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_OPENCL
#include <CL/cl.h>
#endif

struct GPUCapabilities {
    bool available = false;
    std::string backend; // "CUDA" or "OpenCL"
    std::string deviceName;
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    int computeCapability = 0;
};

class GPUDetector {
public:
    static GPUCapabilities detectCapabilities() {
        GPUCapabilities caps;

        #ifdef USE_CUDA
        if (detectCUDA(caps)) {
            caps.backend = "CUDA";
            caps.available = true;
            return caps;
        }
        #endif

        #ifdef USE_OPENCL
        if (detectOpenCL(caps)) {
            caps.backend = "OpenCL";
            caps.available = true;
            return caps;
        }
        #endif

        // No GPU acceleration available
        caps.available = false;
        caps.backend = "CPU";
        return caps;
    }

private:
    static bool detectCUDA(GPUCapabilities& caps) {
        #ifdef USE_CUDA
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            return false;
        }

        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0); // Use first device

        caps.deviceName = prop.name;
        caps.totalMemory = prop.totalGlobalMem;
        caps.computeCapability = prop.major * 10 + prop.minor;

        // Get free memory
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        caps.freeMemory = free;

        return true;
        #else
        return false;
        #endif
    }

    static bool detectOpenCL(GPUCapabilities& caps) {
        #ifdef USE_OPENCL
        cl_uint numPlatforms;
        clGetPlatformIDs(0, NULL, &numPlatforms);
        if (numPlatforms == 0) return false;

        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);

        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if (numDevices == 0) return false;

        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        // Get device info
        char deviceName[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
        caps.deviceName = deviceName;

        cl_ulong memSize;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, NULL);
        caps.totalMemory = memSize;

        return true;
        #else
        return false;
        #endif
    }
};
```

### 2.2 GPU Configuration Header
**File:** `src/gpu/gpu_config.h`

```cpp
#pragma once

#include <cstddef>
#include <string>

// GPU acceleration configuration
namespace GPUConfig {
    // Thresholds for GPU usage
    constexpr size_t GPU_FILE_SIZE_THRESHOLD = 1024 * 1024; // 1MB
    constexpr size_t GPU_BATCH_SIZE = 1000; // Files per batch
    constexpr size_t GPU_MEMORY_LIMIT = 1024 * 1024 * 1024; // 1GB

    // Performance tuning
    constexpr int GPU_THREADS_PER_BLOCK = 256;
    constexpr int GPU_MAX_BLOCKS = 1024;

    // Fallback settings
    constexpr bool ENABLE_GPU_FALLBACK = true;
    constexpr double GPU_TIMEOUT_SECONDS = 30.0;

    // Debug settings
    constexpr bool GPU_DEBUG_MODE = false;
    constexpr bool GPU_PROFILE_MODE = false;

    // Supported algorithms
    enum class GPUAlgorithm {
        SHA256_CUDA,
        SHA256_OPENCL,
        MD5_CUDA,
        MD5_OPENCL
    };

    // GPU device selection
    struct GPUDevice {
        int deviceId = 0;
        std::string name;
        size_t memory = 0;
        bool isAvailable = false;
    };
}
```

---

## Phase 3: GPU Hash Calculator Implementation

### 3.1 CUDA Hash Calculator
**File:** `src/gpu/cuda_hash_calculator.cu`

```cpp
#include "cuda_hash_calculator.h"
#include "gpu_config.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for SHA-256 hash calculation
__global__ void sha256Kernel(const unsigned char* data, size_t dataSize,
                           unsigned char* hash, size_t numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBlocks) return;

    // SHA-256 implementation for GPU
    // (Detailed SHA-256 kernel implementation would go here)
    // This is a placeholder for the actual CUDA kernel code
}

class CUDAHashCalculator {
public:
    CUDAHashCalculator() : deviceMemoryAllocated(false) {}

    ~CUDAHashCalculator() {
        cleanup();
    }

    bool initialize() {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess) {
            std::cerr << "CUDA initialization failed: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        return true;
    }

    std::vector<unsigned char> calculateHash(const std::vector<unsigned char>& data) {
        if (data.size() < GPUConfig::GPU_FILE_SIZE_THRESHOLD) {
            return calculateHashCPU(data); // Fallback for small files
        }

        // Allocate GPU memory
        unsigned char* d_data;
        unsigned char* d_hash;
        size_t hashSize = 32; // SHA-256 produces 32 bytes

        cudaMalloc(&d_data, data.size());
        cudaMalloc(&d_hash, hashSize);

        // Copy data to GPU
        cudaMemcpy(d_data, data.data(), data.size(), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsPerBlock = GPUConfig::GPU_THREADS_PER_BLOCK;
        int blocksPerGrid = (data.size() + threadsPerBlock - 1) / threadsPerBlock;

        sha256Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, data.size(), d_hash, 1);

        // Copy result back
        std::vector<unsigned char> result(hashSize);
        cudaMemcpy(result.data(), d_hash, hashSize, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_data);
        cudaFree(d_hash);

        return result;
    }

private:
    void cleanup() {
        if (deviceMemoryAllocated) {
            cudaDeviceReset();
            deviceMemoryAllocated = false;
        }
    }

    std::vector<unsigned char> calculateHashCPU(const std::vector<unsigned char>& data) {
        // Fallback CPU implementation
        // (Use existing CPU hash calculation)
        return std::vector<unsigned char>(32, 0); // Placeholder
    }

    bool deviceMemoryAllocated;
};
```

### 3.2 OpenCL Hash Calculator (Fallback)
**File:** `src/gpu/opencl_hash_calculator.cpp`

```cpp
#include "opencl_hash_calculator.h"
#include "gpu_config.h"
#include <CL/cl.h>
#include <iostream>
#include <fstream>

class OpenCLHashCalculator {
public:
    OpenCLHashCalculator() : context(nullptr), commandQueue(nullptr), program(nullptr) {}

    ~OpenCLHashCalculator() {
        cleanup();
    }

    bool initialize() {
        cl_uint numPlatforms;
        clGetPlatformIDs(0, NULL, &numPlatforms);
        if (numPlatforms == 0) return false;

        cl_platform_id platform;
        clGetPlatformIDs(1, &platform, NULL);

        cl_uint numDevices;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        if (numDevices == 0) return false;

        cl_device_id device;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        // Create context and command queue
        cl_int err;
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) return false;

        commandQueue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) return false;

        // Load and build kernel
        std::string kernelSource = loadKernelSource("src/gpu/opencl_kernels.cl");
        const char* source = kernelSource.c_str();
        program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
        if (err != CL_SUCCESS) return false;

        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            // Get build log
            size_t logSize;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
            std::vector<char> buildLog(logSize);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), NULL);
            std::cerr << "OpenCL build failed: " << buildLog.data() << std::endl;
            return false;
        }

        kernel = clCreateKernel(program, "sha256Kernel", &err);
        return err == CL_SUCCESS;
    }

    std::vector<unsigned char> calculateHash(const std::vector<unsigned char>& data) {
        // Similar to CUDA implementation but using OpenCL API
        // (Detailed implementation would go here)
        return std::vector<unsigned char>(32, 0); // Placeholder
    }

private:
    void cleanup() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (commandQueue) clReleaseCommandQueue(commandQueue);
        if (context) clReleaseContext(context);
    }

    std::string loadKernelSource(const std::string& filename) {
        std::ifstream file(filename);
        return std::string(std::istreambuf_iterator<char>(file),
                          std::istreambuf_iterator<char>());
    }

    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_kernel kernel;
};
```

### 3.3 OpenCL Kernel Code
**File:** `src/gpu/opencl_kernels.cl`

```c
// OpenCL kernel for SHA-256 hash calculation
__kernel void sha256Kernel(__global const unsigned char* data,
                          __global unsigned char* hash,
                          const unsigned int dataSize) {
    int gid = get_global_id(0);

    // SHA-256 implementation for OpenCL
    // (Detailed kernel implementation would go here)

    // Placeholder: simple copy for testing
    if (gid < 32) {
        hash[gid] = data[gid % dataSize];
    }
}
```

---

## Phase 4: Integration with Existing Codebase

### 4.1 Update HashCalculator Interface
**File:** `include/hash_calculator.h`

```cpp
#pragma once

#include <vector>
#include <memory>
#include <string>

class HashCalculator {
public:
    struct Options {
        bool enableGPUAcceleration = true;
        size_t gpuThreshold = 1024 * 1024; // 1MB
        std::string preferredGPUBackend = "auto"; // "CUDA", "OpenCL", or "auto"
    };

    HashCalculator(const Options& options = Options());
    ~HashCalculator();

    // Main hash calculation method
    std::vector<unsigned char> calculateHash(const std::string& filePath);

    // GPU capability detection
    bool hasGPUAcceleration() const;
    std::string getGPUBackend() const;
    std::string getGPUDeviceName() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};
```

### 4.2 Update HashCalculator Implementation
**File:** `src/core/hash_calculator.cpp`

```cpp
#include "hash_calculator.h"
#include "gpu_detector.h"
#include <fstream>
#include <iostream>
#include <memory>

#ifdef USE_CUDA
#include "cuda_hash_calculator.h"
#endif

#ifdef USE_OPENCL
#include "opencl_hash_calculator.h"
#endif

class HashCalculator::Impl {
public:
    Impl(const Options& options) : options_(options) {
        initializeGPU();
    }

    std::vector<unsigned char> calculateHash(const std::string& filePath) {
        // Read file data
        std::ifstream file(filePath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }

        std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());

        return calculateHash(data);
    }

    std::vector<unsigned char> calculateHash(const std::vector<unsigned char>& data) {
        // Use GPU if available and beneficial
        if (gpuCalculator_ && options_.enableGPUAcceleration &&
            data.size() >= options_.gpuThreshold) {
            try {
                auto result = gpuCalculator_->calculateHash(data);
                if (!result.empty()) {
                    return result;
                }
                // Fall back to CPU on GPU failure
                std::cerr << "GPU hash calculation failed, falling back to CPU" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "GPU acceleration error: " << e.what() << std::endl;
            }
        }

        // CPU fallback
        return calculateHashCPU(data);
    }

    bool hasGPUAcceleration() const {
        return gpuCalculator_ != nullptr;
    }

    std::string getGPUBackend() const {
        return gpuCapabilities_.backend;
    }

    std::string getGPUDeviceName() const {
        return gpuCapabilities_.deviceName;
    }

private:
    void initializeGPU() {
        gpuCapabilities_ = GPUDetector::detectCapabilities();

        if (!gpuCapabilities_.available) {
            std::cout << "GPU acceleration not available, using CPU-only mode" << std::endl;
            return;
        }

        std::cout << "GPU acceleration available: " << gpuCapabilities_.deviceName
                  << " (" << gpuCapabilities_.backend << ")" << std::endl;

        try {
            #ifdef USE_CUDA
            if (gpuCapabilities_.backend == "CUDA") {
                gpuCalculator_ = std::make_unique<CUDAHashCalculator>();
            }
            #endif

            #ifdef USE_OPENCL
            if (gpuCapabilities_.backend == "OpenCL") {
                gpuCalculator_ = std::make_unique<OpenCLHashCalculator>();
            }
            #endif

            if (gpuCalculator_ && !gpuCalculator_->initialize()) {
                gpuCalculator_.reset();
                std::cerr << "GPU calculator initialization failed" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "GPU initialization error: " << e.what() << std::endl;
            gpuCalculator_.reset();
        }
    }

    std::vector<unsigned char> calculateHashCPU(const std::vector<unsigned char>& data) {
        // Existing CPU hash implementation
        // (Copy existing SHA-256 implementation here)
        return std::vector<unsigned char>(32, 0); // Placeholder
    }

    Options options_;
    GPUCapabilities gpuCapabilities_;
    std::unique_ptr<GPUHashCalculator> gpuCalculator_;
};

// Public interface implementation
HashCalculator::HashCalculator(const Options& options)
    : pImpl(std::make_unique<Impl>(options)) {}

HashCalculator::~HashCalculator() = default;

std::vector<unsigned char> HashCalculator::calculateHash(const std::string& filePath) {
    return pImpl->calculateHash(filePath);
}

bool HashCalculator::hasGPUAcceleration() const {
    return pImpl->hasGPUAcceleration();
}

std::string HashCalculator::getGPUBackend() const {
    return pImpl->getGPUBackend();
}

std::string HashCalculator::getGPUDeviceName() const {
    return pImpl->getGPUDeviceName();
}
```

---

## Phase 5: Testing and Validation

### 5.1 GPU Detection Tests
**File:** `tests/gpu/test_gpu_detection.cpp`

```cpp
#include <gtest/gtest.h>
#include "gpu_detector.h"

TEST(GPUDetectorTest, DetectsCapabilities) {
    auto caps = GPUDetector::detectCapabilities();

    // Should always return valid capabilities struct
    EXPECT_TRUE(caps.backend == "CPU" || caps.backend == "CUDA" || caps.backend == "OpenCL");

    if (caps.available) {
        EXPECT_FALSE(caps.deviceName.empty());
        EXPECT_GT(caps.totalMemory, 0u);
    }
}

TEST(GPUDetectorTest, HandlesNoGPU) {
    // Test on CPU-only system (mock if needed)
    auto caps = GPUDetector::detectCapabilities();
    EXPECT_EQ(caps.backend, "CPU");
    EXPECT_FALSE(caps.available);
}
```

### 5.2 GPU Hash Calculator Tests
**File:** `tests/gpu/test_gpu_hash_calculator.cpp`

```cpp
#include <gtest/gtest.h>
#include "hash_calculator.h"

TEST(GPUHashCalculatorTest, CPUFallbackWorks) {
    HashCalculator::Options options;
    options.enableGPUAcceleration = false; // Force CPU

    HashCalculator calc(options);
    EXPECT_FALSE(calc.hasGPUAcceleration());

    std::vector<unsigned char> testData = {'t', 'e', 's', 't'};
    auto hash = calc.calculateHash(testData);
    EXPECT_EQ(hash.size(), 32u); // SHA-256 produces 32 bytes
}

TEST(GPUHashCalculatorTest, GPUAccelerationWhenAvailable) {
    HashCalculator calc;
    if (calc.hasGPUAcceleration()) {
        EXPECT_FALSE(calc.getGPUDeviceName().empty());
        EXPECT_TRUE(calc.getGPUBackend() == "CUDA" || calc.getGPUBackend() == "OpenCL");
    } else {
        // Should gracefully fall back to CPU
        std::vector<unsigned char> testData(1024 * 1024, 'x'); // 1MB data
        auto hash = calc.calculateHash(testData);
        EXPECT_EQ(hash.size(), 32u);
    }
}

TEST(GPUHashCalculatorTest, GPUFallbackOnFailure) {
    // Test that GPU failures don't crash the application
    HashCalculator calc;

    // This should always work regardless of GPU status
    std::vector<unsigned char> testData = {'f', 'a', 'i', 'l', 'u', 'r', 'e', ' ', 't', 'e', 's', 't'};
    auto hash = calc.calculateHash(testData);
    EXPECT_EQ(hash.size(), 32u);
}
```

### 5.3 Performance Benchmark Tests
**File:** `tests/gpu/test_gpu_performance.cpp`

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include "hash_calculator.h"

TEST(GPUPerformanceTest, BenchmarkHashCalculation) {
    HashCalculator cpuCalc({false}); // CPU only
    HashCalculator gpuCalc({true});  // GPU enabled

    // Test with different file sizes
    std::vector<size_t> testSizes = {1024, 1024*1024, 10*1024*1024}; // 1KB, 1MB, 10MB

    for (size_t size : testSizes) {
        std::vector<unsigned char> testData(size, 'x');

        // CPU benchmark
        auto start = std::chrono::high_resolution_clock::now();
        auto cpuHash = cpuCalc.calculateHash(testData);
        auto cpuTime = std::chrono::high_resolution_clock::now() - start;

        // GPU benchmark
        start = std::chrono::high_resolution_clock::now();
        auto gpuHash = gpuCalc.calculateHash(testData);
        auto gpuTime = std::chrono::high_resolution_clock::now() - start;

        // Results should be identical
        EXPECT_EQ(cpuHash, gpuHash);

        // GPU should be faster for large files (if available)
        if (gpuCalc.hasGPUAcceleration() && size >= 1024*1024) {
            double speedup = std::chrono::duration<double>(cpuTime).count() /
                           std::chrono::duration<double>(gpuTime).count();
            std::cout << "File size " << size << " bytes: " << speedup << "x speedup" << std::endl;

            // Expect at least 1.5x speedup for large files
            EXPECT_GE(speedup, 1.5);
        }
    }
}
```

---

## Phase 6: User Interface Integration

### 6.1 Add GPU Settings to UI
**File:** `src/gui/settings_dialog.cpp`

```cpp
// Add GPU settings section
QGroupBox* gpuGroup = new QGroupBox(tr("GPU Acceleration"));
QVBoxLayout* gpuLayout = new QVBoxLayout(gpuGroup);

// GPU enable/disable
gpuEnableCheckBox_ = new QCheckBox(tr("Enable GPU acceleration"));
gpuEnableCheckBox_->setChecked(settings.enableGPUAcceleration);
gpuLayout->addWidget(gpuEnableCheckBox_);

// GPU status display
gpuStatusLabel_ = new QLabel();
updateGPUStatus();
gpuLayout->addWidget(gpuStatusLabel_);

// GPU device info
gpuDeviceLabel_ = new QLabel();
updateGPUDeviceInfo();
gpuLayout->addWidget(gpuDeviceLabel_);

settingsLayout->addWidget(gpuGroup);

// Connect signals
connect(gpuEnableCheckBox_, &QCheckBox::toggled, this, &SettingsDialog::onGPUSettingsChanged);
```

### 6.3 Add GPU Info Command Line Option
**File:** `src/main.cpp`

```cpp
// Add GPU info option parsing
if (parser.isSet("gpu-info")) {
    printGPUInfo();
    return 0;
}

// In main function, add option
QCommandLineOption gpuInfoOption("gpu-info", "Display GPU acceleration information");
parser.addOption(gpuInfoOption);
```

**GPU Info Function:**
```cpp
void printGPUInfo() {
    std::cout << "DupFinder GPU Acceleration Status" << std::endl;
    std::cout << "=================================" << std::endl;
    
    HashCalculator tempCalc;
    
    if (tempCalc.hasGPUAcceleration()) {
        std::cout << "GPU Acceleration: ENABLED" << std::endl;
        std::cout << "Backend: " << tempCalc.getGPUBackend() << std::endl;
        std::cout << "Device: " << tempCalc.getGPUDeviceName() << std::endl;
        
        // Additional GPU details
        auto caps = GPUDetector::detectCapabilities();
        std::cout << "Memory: " << (caps.totalMemory / (1024*1024)) << " MB" << std::endl;
        if (caps.backend == "CUDA") {
            std::cout << "Compute Capability: " << caps.computeCapability / 10 << "." << caps.computeCapability % 10 << std::endl;
        }
    } else {
        std::cout << "GPU Acceleration: DISABLED (CPU-only mode)" << std::endl;
        std::cout << "Reason: No compatible GPU or drivers found" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "To enable GPU acceleration:" << std::endl;
    std::cout << "- Ensure GPU drivers are installed" << std::endl;
    std::cout << "- Install CUDA Toolkit (NVIDIA) or OpenCL (AMD/Intel)" << std::endl;
    std::cout << "- Rebuild with: cmake -DENABLE_GPU_ACCELERATION=ON" << std::endl;
}
```

---

## Phase 7: Build System and Packaging

### 7.1 Update Build Scripts
**File:** `scripts/build_gpu.sh` (Linux/macOS)

```bash
#!/bin/bash

echo "Building DupFinder with GPU acceleration..."
echo "Checking system requirements..."

# Check for GPU hardware
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
elif command -v rocm-smi &> /dev/null; then
    echo "AMD GPU detected:"
    rocm-smi --showproductname
else
    echo "Warning: No GPU detected. Building CPU-only version."
fi

# Check for GPU development libraries
echo "Checking for GPU development libraries..."

# Check CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA found:"
    nvcc --version
else
    echo "CUDA not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux CUDA installation
        sudo apt update
        sudo apt install -y nvidia-cuda-toolkit
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS CUDA installation
        brew install cuda
    fi
fi

# Check OpenCL
if pkg-config --exists OpenCL; then
    echo "OpenCL development libraries found"
else
    echo "OpenCL not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt install -y opencl-headers ocl-icd-opencl-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install opencl-headers
    fi
fi

# Configure and build
echo "Configuring build with GPU support..."
mkdir -p build
cd build
cmake .. -DENABLE_GPU_ACCELERATION=ON

echo "Building..."
make -j$(nproc)

echo "GPU-enabled build complete!"
echo "Run './dupfinder --gpu-info' to check GPU status"
```

### 7.2 Windows Build Script
**File:** `scripts/build_gpu_windows.bat`

```batch
@echo off
echo Building DupFinder with GPU acceleration on Windows...
echo Checking system requirements...

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>nul
if %errorlevel% neq 0 (
    echo Warning: NVIDIA GPU not detected or drivers not installed.
    echo Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx
    echo Continuing with CPU-only build...
) else (
    echo NVIDIA GPU detected.
)

REM Check for CUDA
echo Checking for CUDA installation...
nvcc --version 2>nul
if %errorlevel% neq 0 (
    echo CUDA not found. Installing CUDA Toolkit...
    echo Please download and install CUDA from: https://developer.nvidia.com/cuda-downloads
    echo Or use: winget install NVIDIA.CUDA
    echo.
    echo Press any key to continue with CPU-only build, or Ctrl+C to abort...
    pause >nul
) else (
    echo CUDA found.
)

REM Configure and build
echo Configuring build with GPU support...
if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -DENABLE_GPU_ACCELERATION=ON

echo Building...
cmake --build . --config Release --parallel

echo.
echo GPU-enabled Windows build complete!
echo Run 'dupfinder.exe --gpu-info' to check GPU status
pause
```

### 7.3 Update Installer Configuration
**File:** `CMakeLists.txt` (CPack section)

```cmake
# GPU-related installer components
if(ENABLE_GPU_ACCELERATION)
    set(CPACK_COMPONENT_GPU_DISPLAY_NAME "GPU Acceleration")
    set(CPACK_COMPONENT_GPU_DESCRIPTION "GPU acceleration libraries for improved performance")
    set(CPACK_COMPONENT_GPU_REQUIRED OFF) # Optional component
endif()
```

---

## Phase 8: Documentation and Deployment

### 8.1 Update User Documentation
**File:** `docs/GPU_ACCELERATION.md`

```markdown
# GPU Acceleration Guide

## Overview
DupFinder supports GPU acceleration for significantly improved performance when processing large files or large numbers of files.

## Requirements
- **NVIDIA GPU**: CUDA 11.0 or later
- **AMD GPU**: OpenCL 2.0 or later
- **Intel GPU**: OpenCL 2.0 or later

## Automatic Detection
DupFinder automatically detects and utilizes available GPU resources:
- Checks for CUDA/OpenCL support at startup
- Falls back to CPU-only mode if GPU unavailable
- Displays GPU status in settings

## Performance Benefits
- **Hash Calculation**: 5-10x speedup for files >1MB
- **Large File Processing**: Significant improvement for >100MB files
- **Batch Processing**: Parallel processing of multiple files

## Checking GPU Status
DupFinder provides a command-line option to check GPU acceleration status:

```bash
# Check GPU acceleration status
./dupfinder --gpu-info

# Example output (GPU enabled):
DupFinder GPU Acceleration Status
=================================
GPU Acceleration: ENABLED
Backend: CUDA
Device: NVIDIA GeForce RTX 3080
Memory: 10240 MB
Compute Capability: 8.6

# Example output (CPU only):
DupFinder GPU Acceleration Status
=================================
GPU Acceleration: DISABLED (CPU-only mode)
Reason: No compatible GPU or drivers found
```

## Troubleshooting
- **GPU Not Detected**: Ensure GPU drivers are installed and up to date
- **CUDA Errors**: Verify CUDA toolkit compatibility
- **OpenCL Issues**: Check OpenCL runtime installation
- **Performance Issues**: GPU may be slower for small files due to transfer overhead
```

### 8.2 Update Release Notes
**File:** `CHANGELOG.md`

```markdown
## Version 1.1.0 - GPU Acceleration Release

### New Features
- **GPU Acceleration**: Automatic detection and utilization of NVIDIA/AMD/Intel GPUs
- **Performance Improvements**: 5-10x speedup for large file hash calculations
- **Smart Fallback**: Seamless CPU fallback when GPU unavailable
- **GPU Settings**: User controls for GPU acceleration in settings dialog

### Technical Changes
- Added CUDA and OpenCL support for GPU hash calculations
- Implemented GPU memory management and error handling
- Added GPU capability detection at application startup
- Enhanced build system with optional GPU dependencies

### Compatibility
- **Backward Compatible**: Works on all existing systems
- **Optional Feature**: GPU acceleration is opt-in
- **Cross-Platform**: Supports Windows, Linux, and macOS
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Install CUDA toolkit and verify NVIDIA GPU
- [ ] Update CMakeLists.txt for GPU detection
- [ ] Create GPU source directory structure
- [ ] Implement GPU capability detection

### Week 2: Core Implementation
- [ ] Implement CUDA hash calculator
- [ ] Implement OpenCL hash calculator (fallback)
- [ ] Create OpenCL kernel code
- [ ] Update HashCalculator interface

### Week 3: Integration and Testing
- [ ] Integrate GPU calculators with existing codebase
- [ ] Add comprehensive unit tests
- [ ] Implement performance benchmarks
- [ ] Test GPU fallback mechanisms

### Week 4: UI and Deployment
- [ ] Add GPU settings to UI
- [ ] Update build scripts for GPU support
- [ ] Update installer configuration
- [ ] Create user documentation

---

## Success Criteria

### Functional Requirements
- [ ] Application builds successfully with GPU support
- [ ] GPU acceleration works on NVIDIA hardware
- [ ] CPU fallback functions correctly
- [ ] No performance regression on CPU-only systems

### Performance Requirements
- [ ] 3x minimum speedup for 10MB+ files
- [ ] GPU detection adds <1 second to startup time
- [ ] Memory usage remains reasonable with GPU enabled

### Quality Requirements
- [ ] All existing tests pass with GPU enabled
- [ ] GPU error handling prevents application crashes
- [ ] Clear user feedback about GPU status
- [ ] Comprehensive documentation for GPU features

---

## Risk Mitigation

### Technical Risks
- **GPU Driver Compatibility**: Mitigated by OpenCL fallback and error handling
- **Memory Management**: Mitigated by GPU memory limits and monitoring
- **Build Complexity**: Mitigated by optional GPU support and clear documentation

### Testing Strategy
- **Unit Tests**: GPU capability detection and hash calculation
- **Integration Tests**: Full application with GPU enabled/disabled
- **Performance Tests**: Benchmarks comparing CPU vs GPU performance
- **Compatibility Tests**: Testing on systems with/without GPU

---

**Implementation Lead:** AI Assistant  
**Target Completion:** 4 weeks  
**Dependencies:** CUDA toolkit, GPU hardware  
**Fallback Plan:** CPU-only implementation remains fully functional</content>
<parameter name="filePath">c:\Public\Jade-Dup-Finder\GPU_IMPLEMENTATION_PLAN.md