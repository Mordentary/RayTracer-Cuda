#pragma once
//#include <memory>
//#include "glm/glm.hpp"

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#define checkCudaErrors(val) 
#else

#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )
//#ifdef _DEBUG
//#else
//#define FAKEINIT
//#define checkCudaErrors(val) 
//#endif
#endif


void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}
