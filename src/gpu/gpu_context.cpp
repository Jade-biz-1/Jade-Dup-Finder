#include "gpu_context.h"
#include "gpu_detector.h"
#include <memory>
#include <stdexcept>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef HAS_OPENCL
#include <CL/cl.h>
#endif

namespace GPU {

// CUDA Context Implementation
#ifdef HAS_CUDA
class CUDAContext : public GPUContext {
public:
    CUDAContext() = default;
    ~CUDAContext() override {
        // Cleanup CUDA resources
        if (initialized_) {
            cudaDeviceReset();
        }
    }

    bool initialize() override {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            return false;
        }

        // Get device properties
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            return false;
        }

        deviceName_ = prop.name;
        totalMemory_ = prop.totalGlobalMem;

        initialized_ = true;
        return true;
    }

    void* allocateMemory(size_t size) override {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        return ptr;
    }

    void freeMemory(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    bool copyToDevice(void* devicePtr, const void* hostPtr, size_t size) override {
        cudaError_t err = cudaMemcpy(devicePtr, hostPtr, size, cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    bool copyToHost(void* hostPtr, const void* devicePtr, size_t size) override {
        cudaError_t err = cudaMemcpy(hostPtr, devicePtr, size, cudaMemcpyDeviceToHost);
        return err == cudaSuccess;
    }

    bool synchronize() override {
        cudaError_t err = cudaDeviceSynchronize();
        return err == cudaSuccess;
    }

    size_t getAvailableMemory() const override {
        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        return freeMem;
    }

    size_t getTotalMemory() const override {
        return totalMemory_;
    }

    std::string getDeviceName() const override {
        return deviceName_;
    }

    std::string getBackendType() const override {
        return "CUDA";
    }

    bool isValid() const override {
        return initialized_;
    }

private:
    bool initialized_ = false;
    std::string deviceName_;
    size_t totalMemory_ = 0;
};
#endif

// OpenCL Context Implementation
#ifdef HAS_OPENCL
class OpenCLContext : public GPUContext {
public:
    OpenCLContext() = default;
    ~OpenCLContext() override {
        // Cleanup OpenCL resources
        if (commandQueue_) {
            clReleaseCommandQueue(commandQueue_);
        }
        if (context_) {
            clReleaseContext(context_);
        }
        if (device_) {
            // Device is released by context
        }
    }

    bool initialize() override {
        cl_int err;

        // Get platforms
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

        // Find first platform with GPU devices
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

            device_ = devices[0];
            platform_ = platform;
            break;
        }

        if (!device_) {
            return false;
        }

        // Create context
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Create command queue
        commandQueue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Get device info
        size_t nameSize = 0;
        clGetDeviceInfo(device_, CL_DEVICE_NAME, 0, nullptr, &nameSize);
        std::vector<char> nameBuffer(nameSize);
        clGetDeviceInfo(device_, CL_DEVICE_NAME, nameSize, nameBuffer.data(), nullptr);
        deviceName_ = std::string(nameBuffer.begin(), nameBuffer.end() - 1);

        clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &totalMemory_, nullptr);

        initialized_ = true;
        return true;
    }

    void* allocateMemory(size_t size) override {
        cl_int err;
        cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("OpenCL memory allocation failed");
        }
        return mem;
    }

    void freeMemory(void* ptr) override {
        if (ptr) {
            clReleaseMemObject(static_cast<cl_mem>(ptr));
        }
    }

    bool copyToDevice(void* devicePtr, const void* hostPtr, size_t size) override {
        cl_int err = clEnqueueWriteBuffer(commandQueue_, static_cast<cl_mem>(devicePtr),
                                        CL_TRUE, 0, size, hostPtr, 0, nullptr, nullptr);
        return err == CL_SUCCESS;
    }

    bool copyToHost(void* hostPtr, const void* devicePtr, size_t size) override {
        cl_int err = clEnqueueReadBuffer(commandQueue_, static_cast<cl_mem>(devicePtr),
                                       CL_TRUE, 0, size, hostPtr, 0, nullptr, nullptr);
        return err == CL_SUCCESS;
    }

    bool synchronize() override {
        cl_int err = clFinish(commandQueue_);
        return err == CL_SUCCESS;
    }

    size_t getAvailableMemory() const override {
        // OpenCL doesn't provide direct free memory info, return total as estimate
        return totalMemory_;
    }

    size_t getTotalMemory() const override {
        return totalMemory_;
    }

    std::string getDeviceName() const override {
        return deviceName_;
    }

    std::string getBackendType() const override {
        return "OpenCL";
    }

    bool isValid() const override {
        return initialized_;
    }

private:
    bool initialized_ = false;
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue commandQueue_ = nullptr;
    std::string deviceName_;
    cl_ulong totalMemory_ = 0;
};
#endif

// Factory function implementations
std::unique_ptr<GPUContext> createCUDAContext() {
#ifdef HAS_CUDA
    return std::make_unique<CUDAContext>();
#else
    return nullptr;
#endif
}

std::unique_ptr<GPUContext> createOpenCLContext() {
#ifdef HAS_OPENCL
    return std::make_unique<OpenCLContext>();
#else
    return nullptr;
#endif
}

std::unique_ptr<GPUContext> createOptimalGPUContext(GPUMode mode) {
    auto caps = GPUDetector::detectCapabilities();
    if (!caps.available) {
        return nullptr;
    }

    if (mode == GPUMode::AUTO) {
        // Prefer CUDA over OpenCL
        if (caps.backend == "CUDA") {
            return createCUDAContext();
        } else if (caps.backend == "OpenCL") {
            return createOpenCLContext();
        }
    } else if (mode == GPUMode::CUDA_ONLY) {
        return createCUDAContext();
    } else if (mode == GPUMode::OPENCL_ONLY) {
        return createOpenCLContext();
    }

    return nullptr;
}

} // namespace GPU