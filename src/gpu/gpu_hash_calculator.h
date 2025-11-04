#pragma once

#include "gpu_config.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace GPU {

// Forward declarations
class GPUContext;

// Base class for GPU-accelerated hash calculators
class GPUHashCalculator {
public:
    virtual ~GPUHashCalculator() = default;

    // Initialize the GPU calculator
    virtual bool initialize() = 0;

    // Compute hash for a single data block
    virtual std::vector<uint8_t> computeHash(const std::vector<uint8_t>& data) = 0;

    // Compute hashes for multiple data blocks (batch processing)
    virtual std::vector<std::vector<uint8_t>> computeHashes(
        const std::vector<std::vector<uint8_t>>& dataBlocks) = 0;

    // Get the hash algorithm name
    virtual std::string getAlgorithmName() const = 0;

    // Get the backend type (CUDA/OpenCL)
    virtual std::string getBackendType() const = 0;

    // Check if the calculator is ready for use
    virtual bool isReady() const = 0;

    // Get performance metrics
    struct PerformanceMetrics {
        double throughputMBps = 0.0;     // MB/s
        double latencyMs = 0.0;          // milliseconds
        size_t totalProcessedBytes = 0;  // total bytes processed
        int operationsCount = 0;         // number of operations
    };
    virtual PerformanceMetrics getPerformanceMetrics() const = 0;

protected:
    std::unique_ptr<GPUContext> context_;
    PerformanceMetrics metrics_;
};

// Factory function to create appropriate GPU hash calculator
std::unique_ptr<GPUHashCalculator> createGPUHashCalculator(
    const std::string& algorithm,
    GPUMode mode = GPUMode::AUTO);

// Utility functions
namespace Utils {
    // Check if GPU acceleration is beneficial for given data size
    bool shouldUseGPU(size_t dataSize);

    // Get optimal block size for GPU processing
    size_t getOptimalBlockSize(const std::string& algorithm);

    // Validate GPU memory availability
    bool validateMemoryAvailability(size_t requiredBytes);
}

} // namespace GPU