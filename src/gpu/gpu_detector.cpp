#include "gpu_detector.h"
#include <iostream>
#include <vector>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

namespace GPU {

GPUCapabilities GPUDetector::detectCapabilities() {
    GPUCapabilities caps;

    // Try CUDA first (preferred for NVIDIA GPUs)
    if (detectCUDA(caps)) {
        caps.backend = "CUDA";
        caps.available = true;
        return caps;
    }

    // Try OpenCL as fallback (works with AMD/Intel GPUs)
    if (detectOpenCL(caps)) {
        caps.backend = "OpenCL";
        caps.available = true;
        return caps;
    }

    // No GPU acceleration available
    caps.backend = "CPU";
    caps.available = false;
    caps.deviceName = "CPU Fallback";
    return caps;
}

bool GPUDetector::detectCUDA(GPUCapabilities& caps) {
#ifdef HAS_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    // Get device properties
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, 0); // Use first device

    if (error != cudaSuccess) {
        return false;
    }

    caps.deviceName = deviceProp.name;
    caps.totalMemory = deviceProp.totalGlobalMem;
    caps.computeCapability = deviceProp.major * 10 + deviceProp.minor;

    // Get free memory
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    caps.freeMemory = freeMem;

    return true;
#else
    return false;
#endif
}

bool GPUDetector::detectOpenCL(GPUCapabilities& caps) {
#ifdef HAS_OPENCL
    cl_int err;
    cl_uint numPlatforms = 0;

    err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }

    // Check each platform for devices
    for (cl_platform_id platform : platforms) {
        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            continue;
        }

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }

        // Use first GPU device found
        cl_device_id device = devices[0];

        // Get device name
        size_t nameSize = 0;
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &nameSize);
        std::vector<char> nameBuffer(nameSize);
        clGetDeviceInfo(device, CL_DEVICE_NAME, nameSize, nameBuffer.data(), nullptr);
        caps.deviceName = std::string(nameBuffer.begin(), nameBuffer.end() - 1); // Remove null terminator

        // Get memory info
        cl_ulong globalMemSize = 0;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, nullptr);
        caps.totalMemory = static_cast<size_t>(globalMemSize);

        // OpenCL doesn't have a direct "free memory" equivalent, but we can estimate
        caps.freeMemory = caps.totalMemory; // Conservative estimate

        return true;
    }

    return false;
#else
    return false;
#endif
}

} // namespace GPU