#include "gpu_hash_calculator.h"
#include "gpu_detector.h"
#include "gpu_config.h"
#include <algorithm>
#include <stdexcept>

#ifdef HAS_CUDA
#include "cuda_hash_calculator.h"
#endif

#ifdef HAS_OPENCL
#include "opencl_hash_calculator.h"
#endif

namespace GPU {

// Factory function implementation
std::unique_ptr<GPUHashCalculator> createGPUHashCalculator(
    const std::string& algorithm,
    GPUMode mode) {

    auto caps = GPUDetector::detectCapabilities();
    if (!caps.available) {
        return nullptr;
    }

    std::unique_ptr<GPUHashCalculator> calculator;

    // Try CUDA first if available and requested
    if ((mode == GPUMode::AUTO || mode == GPUMode::CUDA_ONLY) && caps.backend == "CUDA") {
#ifdef HAS_CUDA
        calculator = std::make_unique<CUDAHashCalculator>();
#endif
    }

    // Try OpenCL as fallback
    if (!calculator && (mode == GPUMode::AUTO || mode == GPUMode::OPENCL_ONLY) && caps.backend == "OpenCL") {
#ifdef HAS_OPENCL
        calculator = std::make_unique<OpenCLHashCalculator>();
#endif
    }

    // Initialize the calculator
    if (calculator && !calculator->initialize()) {
        return nullptr;
    }

    return calculator;
}

// Utility functions implementation
namespace Utils {

bool shouldUseGPU(size_t dataSize) {
    // Only use GPU for sufficiently large data
    return dataSize >= Config::MIN_GPU_FILE_SIZE;
}

size_t getOptimalBlockSize(const std::string& algorithm) {
    if (algorithm == "MD5") {
        return Config::MD5_BLOCK_SIZE;
    } else if (algorithm == "SHA256") {
        return Config::SHA256_BLOCK_SIZE;
    } else if (algorithm == "Perceptual") {
        return Config::PERCEPTUAL_BLOCK_SIZE;
    }
    return Config::MD5_BLOCK_SIZE; // Default
}

bool validateMemoryAvailability(size_t requiredBytes) {
    auto caps = GPUDetector::detectCapabilities();
    if (!caps.available) {
        return false;
    }

    // Reserve some memory for GPU operations
    size_t availableBytes = static_cast<size_t>(
        caps.freeMemory * (1.0f - Config::MEMORY_THRESHOLD_WARNING));

    return requiredBytes <= availableBytes;
}

} // namespace Utils

} // namespace GPU