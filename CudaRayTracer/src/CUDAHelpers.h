#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(val) CUDAHelpers::checkCuda( (val), #val, __FILE__, __LINE__ )
#define COPY_TO_SYMBOL(smbl, src, size) cudaMemcpyToSymbol(smbl, src, size)

namespace CUDAHelpers {
	struct RenderConfig {
		dim3 threads;
		dim3 blocks;
		int totalThreads;
	};

	inline void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
		if (result) {
			std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
				<< file << ":" << line << " '" << func << "' \n";
			cudaDeviceReset();
			throw std::runtime_error("CUDA error");
		}
	}

	constexpr int THREADS_PER_BLOCK = 16;
	inline RenderConfig createRenderConfig(int width, int height) {
		RenderConfig config;
		config.threads = dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
		config.blocks = dim3((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (height + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		config.totalThreads = config.blocks.x * config.blocks.y * config.threads.x * config.threads.y;
		return config;
	}
}  // namespace CUDAHelpers
