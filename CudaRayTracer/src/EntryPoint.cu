#include <SFML/Graphics.hpp>
#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>


#include "Raytracer.h"
#include "CUDAHelpers.h"
#include "SceneManager.h"
#include "WindowManager.h"


int main() {
	try {
		constexpr float ASPECT_RATIO = 16.0f / 9.0f;
		constexpr int WIDTH = 1440;
		constexpr int HEIGHT = static_cast<int>(WIDTH / ASPECT_RATIO);
		constexpr float VERTICAL_FOV = 70.0f;
		constexpr float APERTURE = 0.0001f;

		// Increase memory limits
		//size_t heapSize = 20000000 * sizeof(double) * 4;
		//size_t stackSize = 12928;
		//CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));
		//CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));

		size_t size_heap, size_stack;
		CUDA_CHECK(cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize));
		CUDA_CHECK(cudaDeviceGetLimit(&size_stack, cudaLimitStackSize));
		std::cout << "Heap size set to " << size_heap << "; Stack size set to " << size_stack << std::endl;

		Raytracer raytracer(WIDTH, HEIGHT, ASPECT_RATIO, VERTICAL_FOV, APERTURE);
		raytracer.run();

		return EXIT_SUCCESS;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
}