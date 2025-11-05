#include "cuda_hash_calculator.h"
#include "gpu_context.h"  // Include complete GPUContext definition

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

namespace GPU {

// Simple CUDA kernel for demonstration
__global__ void simple_sha256_kernel(const uint8_t* input, size_t inputSize, uint8_t* output) {
    // Dummy implementation for compilation - replace with real SHA-256
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 32) {
        output[idx] = (idx < inputSize) ? input[idx] : 0;
    }
}

class CUDAHashCalculator::Impl {
public:
    Impl() : initialized_(false) {}

    bool initialize() {
        if (initialized_) return true;

        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    void cleanup() {
        if (initialized_) {
            cudaDeviceReset();
            initialized_ = false;
        }
    }

    std::vector<uint8_t> computeHash(const std::vector<uint8_t>& data) {
        if (!initialized_) {
            throw std::runtime_error("CUDA not initialized");
        }

        std::vector<uint8_t> hash(32, 0);

        // Allocate device memory
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;

        cudaError_t err = cudaMalloc(&d_input, data.size());
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate input buffer");
        }

        err = cudaMalloc(&d_output, 32);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            throw std::runtime_error("Failed to allocate output buffer");
        }

        // Copy input data to device
        err = cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Failed to copy input data");
        }

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (32 + threadsPerBlock - 1) / threadsPerBlock;
        simple_sha256_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, data.size(), d_output);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Kernel launch failed");
        }

        // Copy result back
        err = cudaMemcpy(hash.data(), d_output, 32, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Failed to copy result");
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);

        return hash;
    }

    bool isReady() const {
        return initialized_;
    }

private:
    bool initialized_;
};

// CUDAHashCalculator implementation
CUDAHashCalculator::CUDAHashCalculator() : impl_(std::make_unique<Impl>()) {}

CUDAHashCalculator::~CUDAHashCalculator() {
    if (impl_) {
        impl_->cleanup();
    }
}

bool CUDAHashCalculator::initialize() {
    return impl_->initialize();
}

std::vector<uint8_t> CUDAHashCalculator::computeHash(const std::vector<uint8_t>& data) {
    return impl_->computeHash(data);
}

std::vector<std::vector<uint8_t>> CUDAHashCalculator::computeHashes(
    const std::vector<std::vector<uint8_t>>& dataBlocks) {
    std::vector<std::vector<uint8_t>> hashes;
    hashes.reserve(dataBlocks.size());
    for (const auto& data : dataBlocks) {
        hashes.push_back(computeHash(data));
    }
    return hashes;
}

std::string CUDAHashCalculator::getAlgorithmName() const {
    return "SHA-256";
}

std::string CUDAHashCalculator::getBackendType() const {
    return "CUDA";
}

bool CUDAHashCalculator::isReady() const {
    return impl_->isReady();
}

GPUHashCalculator::PerformanceMetrics CUDAHashCalculator::getPerformanceMetrics() const {
    PerformanceMetrics metrics;
    metrics.throughputMBps = 0.0;
    metrics.latencyMs = 0.0;
    metrics.totalProcessedBytes = 0;
    metrics.operationsCount = 0;
    return metrics;
}

} // namespace GPU
