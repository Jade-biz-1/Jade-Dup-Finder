#include "opencl_hash_calculator.h"
#include "gpu_context.h"
#include <CL/cl.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <sstream>

namespace GPU {

// OpenCL kernel source for SHA-256
static const char* sha256_kernel_source = R"(
__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

uint rightRotate(uint value, uint amount) {
    return (value >> amount) | (value << (32 - amount));
}

void sha256_transform(__private uint* state, __private uint* w) {
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint S1 = rightRotate(e, 6) ^ rightRotate(e, 11) ^ rightRotate(e, 25);
        uint ch = (e & f) ^ (~e & g);
        uint temp1 = h + S1 + ch + K[i] + w[i];
        uint S0 = rightRotate(a, 2) ^ rightRotate(a, 7) ^ rightRotate(a, 13);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__kernel void sha256_kernel(__global uchar* data, uint dataSize, __global uint* hashOutput, uint numBlocks) {
    uint idx = get_global_id(0);
    if (idx >= numBlocks) return;

    // SHA-256 initial hash values
    uint state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Prepare message schedule from input data
    uint w[64] = {0};
    __global uchar* chunk = data + idx * 64;

    // Load 512-bit chunk
    for (int i = 0; i < 16; ++i) {
        w[i] = ((uint)chunk[i*4] << 24) | ((uint)chunk[i*4+1] << 16) |
               ((uint)chunk[i*4+2] << 8) | (uint)chunk[i*4+3];
    }

    // Extend to 64 words
    for (int i = 16; i < 64; ++i) {
        uint s0 = rightRotate(w[i-15], 7) ^ rightRotate(w[i-15], 18) ^ (w[i-15] >> 3);
        uint s1 = rightRotate(w[i-2], 17) ^ rightRotate(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // Transform
    sha256_transform(state, w);

    // Store result
    for (int i = 0; i < 8; ++i) {
        hashOutput[idx * 8 + i] = state[i];
    }
}
)";

// Implementation class
class OpenCLHashCalculator::Impl {
public:
    Impl() : context_(createOpenCLContext()), initialized_(false),
             kernel_(nullptr), program_(nullptr), commandQueue_(nullptr) {}

    ~Impl() {
        cleanup();
    }

    bool initialize() {
        if (!context_ || !context_->initialize()) {
            return false;
        }

        // Get OpenCL context internals (this is a simplified approach)
        // In a real implementation, you'd need to expose more from the OpenCL context
        // For now, we'll recreate the OpenCL setup

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

        // Create context and command queue
        contextCL_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        commandQueue_ = clCreateCommandQueueWithProperties(contextCL_, device_, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Create program
        const char* source = sha256_kernel_source;
        program_ = clCreateProgramWithSource(contextCL_, 1, &source, nullptr, &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        // Build program
        err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Get build log for debugging
            size_t logSize = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> buildLog(logSize);
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
            return false;
        }

        // Create kernel
        kernel_ = clCreateKernel(program_, "sha256_kernel", &err);
        if (err != CL_SUCCESS) {
            return false;
        }

        initialized_ = true;
        return true;
    }

    bool isAvailable() const {
        return initialized_;
    }

    HashResult computeHash(const std::vector<uint8_t>& data) {
        if (!isAvailable() || data.empty()) {
            return HashResult{};
        }

        std::vector<std::vector<uint8_t>> batch = {data};
        auto results = computeHashes(batch);
        return results.empty() ? HashResult{} : results[0];
    }

    std::vector<HashResult> computeHashes(const std::vector<std::vector<uint8_t>>& dataBatch) {
        if (!isAvailable() || dataBatch.empty()) {
            return {};
        }

        std::vector<HashResult> results;
        results.reserve(dataBatch.size());

        // Calculate total data size
        size_t totalDataSize = 0;
        for (const auto& data : dataBatch) {
            totalDataSize += data.size();
        }

        cl_int err;
        cl_mem d_data = nullptr;
        cl_mem d_hashOutput = nullptr;

        try {
            // Create buffers
            d_data = clCreateBuffer(contextCL_, CL_MEM_READ_ONLY, totalDataSize, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to create data buffer");
            }

            d_hashOutput = clCreateBuffer(contextCL_, CL_MEM_WRITE_ONLY,
                                        dataBatch.size() * 8 * sizeof(cl_uint), nullptr, &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to create output buffer");
            }

            // Copy data to device
            size_t offset = 0;
            for (const auto& data : dataBatch) {
                err = clEnqueueWriteBuffer(commandQueue_, d_data, CL_FALSE, offset,
                                         data.size(), data.data(), 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    throw std::runtime_error("Failed to write data to buffer");
                }
                offset += data.size();
            }

            // Set kernel arguments
            err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &d_data);
            err |= clSetKernelArg(kernel_, 1, sizeof(cl_uint), &totalDataSize);
            err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &d_hashOutput);
            err |= clSetKernelArg(kernel_, 3, sizeof(cl_uint), &dataBatch.size());

            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to set kernel arguments");
            }

            // Execute kernel
            size_t globalWorkSize = dataBatch.size();
            size_t localWorkSize = 64; // Adjust based on device capabilities

            err = clEnqueueNDRangeKernel(commandQueue_, kernel_, 1, nullptr,
                                       &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to enqueue kernel");
            }

            // Read results
            std::vector<cl_uint> hostHashes(dataBatch.size() * 8);
            err = clEnqueueReadBuffer(commandQueue_, d_hashOutput, CL_TRUE, 0,
                                    hostHashes.size() * sizeof(cl_uint), hostHashes.data(),
                                    0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to read results");
            }

            // Convert to HashResult format
            for (size_t i = 0; i < dataBatch.size(); ++i) {
                HashResult result;
                result.algorithm = "SHA256";
                result.hash.resize(32);

                for (int j = 0; j < 8; ++j) {
                    cl_uint val = hostHashes[i * 8 + j];
                    result.hash[j * 4] = (val >> 24) & 0xFF;
                    result.hash[j * 4 + 1] = (val >> 16) & 0xFF;
                    result.hash[j * 4 + 2] = (val >> 8) & 0xFF;
                    result.hash[j * 4 + 3] = val & 0xFF;
                }

                results.push_back(std::move(result));
            }

        } catch (const std::exception&) {
            results.clear();
        }

        // Cleanup
        if (d_data) clReleaseMemObject(d_data);
        if (d_hashOutput) clReleaseMemObject(d_hashOutput);

        return results;
    }

private:
    void cleanup() {
        if (kernel_) clReleaseKernel(kernel_);
        if (program_) clReleaseProgram(program_);
        if (commandQueue_) clReleaseCommandQueue(commandQueue_);
        if (contextCL_) clReleaseContext(contextCL_);
    }

    std::unique_ptr<GPUContext> context_;
    bool initialized_;

    cl_platform_id platform_;
    cl_device_id device_;
    cl_context contextCL_;
    cl_command_queue commandQueue_;
    cl_program program_;
    cl_kernel kernel_;
};

// OpenCLHashCalculator implementation
OpenCLHashCalculator::OpenCLHashCalculator()
    : impl_(std::make_unique<Impl>()) {}

OpenCLHashCalculator::~OpenCLHashCalculator() = default;

bool OpenCLHashCalculator::initialize() {
    return impl_->initialize();
}

bool OpenCLHashCalculator::isAvailable() const {
    return impl_->isAvailable();
}

std::string OpenCLHashCalculator::getBackendName() const {
    return "OpenCL";
}

HashResult OpenCLHashCalculator::computeHash(const std::vector<uint8_t>& data) {
    return impl_->computeHash(data);
}

std::vector<HashResult> OpenCLHashCalculator::computeHashes(const std::vector<std::vector<uint8_t>>& dataBatch) {
    return impl_->computeHashes(dataBatch);
}

} // namespace GPU