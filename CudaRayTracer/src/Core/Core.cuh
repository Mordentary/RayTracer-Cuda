#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#define CUDA_CHECK(val)
#else

#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#define CUDA_CHECK(val) checkCuda( (val), #val, __FILE__, __LINE__ )
#endif

template<typename T>
class CudaMemory {
private:
	T* ptr;
	size_t size;
	bool isManaged;

public:
	CudaMemory(size_t elementCount, bool globalMem = false)
		: ptr(nullptr), size(elementCount * sizeof(T)), isManaged(globalMem)
	{
		if (!globalMem)
			CUDA_CHECK(cudaMalloc(&ptr, size));
		else
			CUDA_CHECK(cudaMallocManaged(&ptr, size));
	}

	~CudaMemory() {
		if (ptr) CUDA_CHECK(cudaFree(ptr));
	}
	// Delete copy constructor and assignment operator
	CudaMemory(const CudaMemory&) = delete;
	CudaMemory& operator=(const CudaMemory&) = delete;

	CudaMemory(CudaMemory&& other) noexcept
		: ptr(other.ptr), size(other.size), isManaged(other.isManaged)
	{
		other.ptr = nullptr;
		other.size = 0;
	}

	CudaMemory& operator=(CudaMemory&& other) noexcept {
		if (this != &other) {
			if (ptr) CUDA_CHECK(cudaFree(ptr));
			ptr = other.ptr;
			size = other.size;
			isManaged = other.isManaged;
			other.ptr = nullptr;
			other.size = 0;
		}
		return *this;
	}

	T* get() const { return ptr; }
	operator T* () const { return ptr; }
	size_t getSize() const { return size; }
	bool isManagedMemory() const { return isManaged; }
};

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}