#pragma once

#include <string>
#include <cstddef>

namespace GPU {

// GPU capability detection and information
struct GPUCapabilities {
    bool available = false;
    std::string backend; // "CUDA", "OpenCL", or "CPU"
    std::string deviceName;
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    int computeCapability = 0; // For CUDA devices
};

// GPU detector class
class GPUDetector {
public:
    // Detect GPU capabilities on the system
    static GPUCapabilities detectCapabilities();

private:
    // CUDA detection
    static bool detectCUDA(GPUCapabilities& caps);

    // OpenCL detection
    static bool detectOpenCL(GPUCapabilities& caps);
};

} // namespace GPU