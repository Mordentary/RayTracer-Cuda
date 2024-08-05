#pragma once

#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#define CUDA_CHECK(val)
#define COPY_TO_SYMBOL(val)

#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#define CUDA_CHECK(val) checkCuda( (val), #val, __FILE__, __LINE__ )
#define COPY_TO_SYMBOL(smbl, src, size) cudaMemcpyToSymbol(smbl, src, size);


#endif

template<typename T>
class CudaMemory {
private:
	T* m_MemPtr;
	size_t m_Size;
	bool m_IsManaged;

public:
	CudaMemory(size_t size, bool globalMem = false)
		: m_MemPtr(nullptr), m_Size(size), m_IsManaged(globalMem)
	{
		if (!globalMem)
			CUDA_CHECK(cudaMalloc(&m_MemPtr, m_Size));
		else
			CUDA_CHECK(cudaMallocManaged(&m_MemPtr, m_Size));
	}

	~CudaMemory() {
		if (m_MemPtr) CUDA_CHECK(cudaFree(m_MemPtr));
	}
	// Delete copy constructor and assignment operator
	CudaMemory(const CudaMemory&) = delete;
	CudaMemory& operator=(const CudaMemory&) = delete;

	CudaMemory(CudaMemory&& other) noexcept
		: m_MemPtr(other.m_MemPtr), m_Size(other.m_Size), m_IsManaged(other.m_IsManaged)
	{
		other.m_MemPtr = nullptr;
		other.m_Size = 0;
	}

	CudaMemory& operator=(CudaMemory&& other) noexcept {
		if (this != &other) {
			if (m_MemPtr) CUDA_CHECK(cudaFree(m_MemPtr));
			m_MemPtr = other.m_MemPtr;
			m_Size = other.m_Size;
			m_IsManaged = other.m_IsManaged;
			other.m_MemPtr = nullptr;
			other.m_Size = 0;
		}
		return *this;
	}

	T* get() const { return m_MemPtr; }
	operator T* () const { return m_MemPtr; }
	size_t size() const { return m_Size; }
	bool isManagedMemory() const { return m_IsManaged; }
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