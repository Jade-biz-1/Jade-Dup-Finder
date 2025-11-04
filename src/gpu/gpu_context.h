#pragma once

#include "gpu_config.h"
#include <memory>
#include <string>
#include <cstddef>

namespace GPU {

// GPU context interface for managing GPU resources
class GPUContext {
public:
    virtual ~GPUContext() = default;

    // Initialize the GPU context
    virtual bool initialize() = 0;

    // Allocate GPU memory
    virtual void* allocateMemory(size_t size) = 0;

    // Free GPU memory
    virtual void freeMemory(void* ptr) = 0;

    // Copy data from host to device
    virtual bool copyToDevice(void* devicePtr, const void* hostPtr, size_t size) = 0;

    // Copy data from device to host
    virtual bool copyToHost(void* hostPtr, const void* devicePtr, size_t size) = 0;

    // Synchronize GPU operations
    virtual bool synchronize() = 0;

    // Get available memory
    virtual size_t getAvailableMemory() const = 0;

    // Get total memory
    virtual size_t getTotalMemory() const = 0;

    // Get device name
    virtual std::string getDeviceName() const = 0;

    // Get backend type
    virtual std::string getBackendType() const = 0;

    // Check if context is valid
    virtual bool isValid() const = 0;
};

// Factory functions for creating GPU contexts
std::unique_ptr<GPUContext> createCUDAContext();
std::unique_ptr<GPUContext> createOpenCLContext();

// Get the best available GPU context
std::unique_ptr<GPUContext> createOptimalGPUContext(GPUMode mode = GPUMode::AUTO);

} // namespace GPU