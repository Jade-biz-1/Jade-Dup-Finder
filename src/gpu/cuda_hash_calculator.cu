#include "cuda_hash_calculator.h"
#include "gpu_context.h"  // Include complete GPUContext definition

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <chrono>
#include <string>
#include <vector>

namespace GPU {

// SHA-256 constants
__constant__ uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 initial hash values
__constant__ uint32_t h0_init[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Helper functions for SHA-256
__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// CUDA kernel for SHA-256 computation
__global__ void sha256_kernel(const uint8_t* input, size_t inputSize, uint8_t* output) {
    int idx = blockIdx.x;
    if (idx > 0) return; // Only process first block for now (single file)

    // SHA-256 processing for one file
    const uint8_t* data = input;
    size_t dataLen = inputSize;

    // Initialize hash values
    uint32_t h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = h0_init[i];
    }

    // Pre-processing: add padding
    uint64_t bitLen = dataLen * 8;
    size_t paddedLen = ((dataLen + 8) / 64 + 1) * 64;
    uint8_t* padded = new uint8_t[paddedLen];
    memset(padded, 0, paddedLen);
    memcpy(padded, data, dataLen);
    padded[dataLen] = 0x80; // Append '1' bit

    // Append original length as 64-bit big-endian
    for (int i = 0; i < 8; i++) {
        padded[paddedLen - 8 + i] = (bitLen >> (56 - i * 8)) & 0xFF;
    }

    // Process 512-bit chunks
    for (size_t chunk = 0; chunk < paddedLen; chunk += 64) {
        uint32_t w[64];
        uint32_t a, b, c, d, e, f, g, hh;

        // Prepare message schedule
        for (int i = 0; i < 16; i++) {
            w[i] = (padded[chunk + i*4] << 24) |
                   (padded[chunk + i*4 + 1] << 16) |
                   (padded[chunk + i*4 + 2] << 8) |
                   padded[chunk + i*4 + 3];
        }

        for (int i = 16; i < 64; i++) {
            w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
        }

        // Initialize working variables
        a = h[0]; b = h[1]; c = h[2]; d = h[3];
        e = h[4]; f = h[5]; g = h[6]; hh = h[7];

        // Main compression loop
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + sigma1(e) + ch(e, f, g) + k[i] + w[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        // Add compressed chunk to hash
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    delete[] padded;

    // Produce final hash (big-endian)
    for (int i = 0; i < 8; i++) {
        output[i*4] = (h[i] >> 24) & 0xFF;
        output[i*4 + 1] = (h[i] >> 16) & 0xFF;
        output[i*4 + 2] = (h[i] >> 8) & 0xFF;
        output[i*4 + 3] = h[i] & 0xFF;
    }
}

class CUDAHashCalculator::Impl {
public:
    Impl() : initialized_(false), totalProcessedBytes_(0), operationsCount_(0) {}

    bool initialize() {
        if (initialized_) return true;

        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            lastError_ = std::string("Failed to set CUDA device: ") + cudaGetErrorString(err);
            return false;
        }

        // Get device properties for optimization
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            lastError_ = std::string("Failed to get device properties: ") + cudaGetErrorString(err);
            return false;
        }

        deviceName_ = prop.name;
        totalGlobalMem_ = prop.totalGlobalMem;
        multiProcessorCount_ = prop.multiProcessorCount;

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

        auto startTime = std::chrono::high_resolution_clock::now();

        if (data.empty()) {
            // SHA-256 of empty string
            std::vector<uint8_t> hash(32);
            // Precomputed SHA-256 of empty string
            uint32_t empty_hash[8] = {
                0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924,
                0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855
            };
            for (int i = 0; i < 8; i++) {
                hash[i*4] = (empty_hash[i] >> 24) & 0xFF;
                hash[i*4 + 1] = (empty_hash[i] >> 16) & 0xFF;
                hash[i*4 + 2] = (empty_hash[i] >> 8) & 0xFF;
                hash[i*4 + 3] = empty_hash[i] & 0xFF;
            }

            operationsCount_++;
            totalProcessedBytes_ += data.size();

            auto endTime = std::chrono::high_resolution_clock::now();
            lastOperationTimeMs_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();

            return hash;
        }

        std::vector<uint8_t> hash(32, 0);

        // Allocate device memory
        uint8_t* d_input = nullptr;
        uint8_t* d_output = nullptr;

        cudaError_t err;

        err = cudaMalloc(&d_input, data.size());
        if (err != cudaSuccess) {
            lastError_ = std::string("Failed to allocate input buffer: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        err = cudaMalloc(&d_output, 32);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            lastError_ = std::string("Failed to allocate output buffer: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        // Copy input data to device
        err = cudaMemcpy(d_input, data.data(), data.size(), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            lastError_ = std::string("Failed to copy input data: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        // Launch kernel (single block for now)
        sha256_kernel<<<1, 1>>>(d_input, data.size(), d_output);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            lastError_ = std::string("Kernel launch failed: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        // Wait for kernel completion
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            lastError_ = std::string("Kernel synchronization failed: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        // Copy result back
        err = cudaMemcpy(hash.data(), d_output, 32, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            lastError_ = std::string("Failed to copy result: ") + cudaGetErrorString(err);
            throw std::runtime_error(lastError_);
        }

        // Cleanup
        cudaFree(d_input);
        cudaFree(d_output);

        operationsCount_++;
        totalProcessedBytes_ += data.size();

        auto endTime = std::chrono::high_resolution_clock::now();
        lastOperationTimeMs_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();

        return hash;
    }

    bool isReady() const {
        return initialized_;
    }

    std::string getLastError() const {
        return lastError_;
    }

    std::string getDeviceName() const {
        return deviceName_;
    }

    size_t getTotalMemory() const {
        return totalGlobalMem_;
    }

    int getMultiProcessorCount() const {
        return multiProcessorCount_;
    }

    size_t getTotalProcessedBytes() const {
        return totalProcessedBytes_;
    }

    size_t getOperationsCount() const {
        return operationsCount_;
    }

    double getLastOperationTimeMs() const {
        return lastOperationTimeMs_;
    }

private:
    bool initialized_;
    std::string lastError_;
    std::string deviceName_;
    size_t totalGlobalMem_;
    int multiProcessorCount_;
    size_t totalProcessedBytes_;
    size_t operationsCount_;
    double lastOperationTimeMs_;
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

    if (impl_->getOperationsCount() > 0) {
        double totalTimeMs = impl_->getLastOperationTimeMs() * impl_->getOperationsCount();
        double totalTimeSec = totalTimeMs / 1000.0;
        double totalMB = static_cast<double>(impl_->getTotalProcessedBytes()) / (1024.0 * 1024.0);

        if (totalTimeSec > 0) {
            metrics.throughputMBps = totalMB / totalTimeSec;
        }

        metrics.latencyMs = impl_->getLastOperationTimeMs();
        metrics.totalProcessedBytes = impl_->getTotalProcessedBytes();
        metrics.operationsCount = impl_->getOperationsCount();
    } else {
        metrics.throughputMBps = 0.0;
        metrics.latencyMs = 0.0;
        metrics.totalProcessedBytes = 0;
        metrics.operationsCount = 0;
    }

    return metrics;
}

} // namespace GPU
