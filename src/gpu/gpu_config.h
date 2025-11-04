#pragma once

#include <cstddef>

namespace GPU {

// GPU configuration constants
namespace Config {
    // Default block sizes for different hash algorithms
    constexpr size_t MD5_BLOCK_SIZE = 64 * 1024;      // 64KB blocks for MD5
    constexpr size_t SHA256_BLOCK_SIZE = 64 * 1024;   // 64KB blocks for SHA256
    constexpr size_t PERCEPTUAL_BLOCK_SIZE = 256 * 1024; // 256KB for perceptual hashing

    // Maximum concurrent GPU operations
    constexpr int MAX_CONCURRENT_OPERATIONS = 4;

    // Memory thresholds (as percentage of total GPU memory)
    constexpr float MEMORY_THRESHOLD_WARNING = 0.8f;  // 80% usage warning
    constexpr float MEMORY_THRESHOLD_CRITICAL = 0.95f; // 95% usage critical

    // Timeout for GPU operations (in milliseconds)
    constexpr int GPU_OPERATION_TIMEOUT_MS = 30000; // 30 seconds

    // Minimum file size for GPU acceleration (bytes)
    constexpr size_t MIN_GPU_FILE_SIZE = 1024 * 1024; // 1MB minimum

    // CUDA-specific settings
    namespace CUDA {
        constexpr int DEFAULT_BLOCK_SIZE = 256;
        constexpr int DEFAULT_GRID_SIZE = 1024;
        constexpr int SHARED_MEMORY_SIZE = 48 * 1024; // 48KB shared memory
    }

    // OpenCL-specific settings
    namespace OpenCL {
        constexpr size_t LOCAL_WORK_SIZE = 256;
        constexpr size_t GLOBAL_WORK_SIZE_MULTIPLIER = 4;
    }
}

// GPU operation modes
enum class GPUMode {
    DISABLED,    // CPU-only operation
    AUTO,        // Automatic GPU selection
    CUDA_ONLY,   // CUDA only
    OPENCL_ONLY, // OpenCL only
    HYBRID       // Use both CUDA and OpenCL if available
};

// GPU memory allocation strategy
enum class MemoryStrategy {
    CONSERVATIVE, // Minimize memory usage
    BALANCED,     // Balance speed and memory
    AGGRESSIVE    // Maximize performance
};

} // namespace GPU