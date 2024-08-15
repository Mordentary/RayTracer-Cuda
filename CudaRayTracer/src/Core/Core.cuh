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

template<class T>
__host__ __device__ void swap(T& val1, T& t2)
{
	T temp = val1;
	val1 = val2;
	val2 = val1;

}

template<class T>
__host__ __device__ void swap_triplet(T& val1, T& val2, T& val3, T& t1, T& t2, T& t3)
{
	T temp1 = val1;
	T temp2 = val2;
	T temp3 = val3;

	val1 = t1;
	val2 = t2;
	val3 = t3;

	t1 = temp1;
	t2 = temp2;
	t3 = temp3;
}

namespace CudaAlloc
{
	enum class Type { Bytes, Elements };

	template<typename T>
	class Memory {
	private:
		T* m_MemPtr;
		size_t m_Size;
		bool m_IsManaged;

	public:
		Memory(size_t sizeOrElements, bool globalMem = false, Type allocType = Type::Elements)
			: m_MemPtr(nullptr), m_IsManaged(globalMem)
		{
			m_Size = (allocType == Type::Elements) ? sizeOrElements * sizeof(T) : sizeOrElements;
			allocateMemory();
		}

		~Memory() {
			if (m_MemPtr) CUDA_CHECK(cudaFree(m_MemPtr));
		}

		// Delete copy constructor and assignment operator
		Memory(const Memory&) = delete;
		Memory& operator=(const Memory&) = delete;

		// Move constructor
		Memory(Memory&& other) noexcept
			: m_MemPtr(other.m_MemPtr), m_Size(other.m_Size), m_IsManaged(other.m_IsManaged)
		{
			other.m_MemPtr = nullptr;
			other.m_Size = 0;
		}

		// Move assignment operator
		Memory& operator=(Memory&& other) noexcept {
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

		void copyFromHost(const T* hostData, size_t count) {
			size_t bytesToCopy = count * sizeof(T);
			if (bytesToCopy > m_Size) {
				std::cerr << "Error: Attempting to copy " << bytesToCopy
					<< " bytes, but only " << m_Size << " bytes allocated." << std::endl;
				throw std::runtime_error("Attempted to copy more data than allocated");
			}
			CUDA_CHECK(cudaMemcpy(m_MemPtr, hostData, bytesToCopy, cudaMemcpyHostToDevice));
		}

		// Copy data from device to host
		void copyToHost(T* hostData, size_t count) const {
			size_t bytesToCopy = count * sizeof(T);
			if (bytesToCopy > m_Size) {
				throw std::runtime_error("Attempted to copy more data than allocated");
			}
			CUDA_CHECK(cudaMemcpy(hostData, m_MemPtr, bytesToCopy, cudaMemcpyDeviceToHost));
		}

		// Set device memory to a specific value
		void memset(int value) {
			CUDA_CHECK(cudaMemset(m_MemPtr, value, m_Size));
		}

		//Cleares memory
		void reallocate(size_t newSize, CudaAlloc::Type allocType = CudaAlloc::Type::Elements) {
			size_t newSizeInBytes = (allocType == CudaAlloc::Type::Elements) ? newSize * sizeof(T) : newSize;

			if (newSizeInBytes == m_Size) return;

			// Allocate new memory
			T* newPtr = nullptr;
			CUDA_CHECK(m_IsManaged ?
				cudaMallocManaged(&newPtr, newSizeInBytes) :
				cudaMalloc(&newPtr, newSizeInBytes));

			if (m_MemPtr) {
				CUDA_CHECK(cudaFree(m_MemPtr));
			}

			m_MemPtr = newPtr;
			m_Size = newSizeInBytes;

		}

		// Synchronize if using managed memory
		void sync() {
			if (m_IsManaged) {
				CUDA_CHECK(cudaDeviceSynchronize());
			}
		}

		// Get a device pointer to a specific element
		T* getElementPtr(size_t index) const {
			if (index * sizeof(T) >= m_Size) {
				throw std::out_of_range("Index out of bounds");
			}
			return m_MemPtr + index;
		}

		T* get() const { return m_MemPtr; }
		operator T* () const { return m_MemPtr; }
		size_t size() const { return m_Size; }
		bool isManagedMemory() const { return m_IsManaged; }

	private:
		void allocateMemory() {
			CUDA_CHECK(m_IsManaged ?
				cudaMallocManaged(&m_MemPtr, m_Size) :
				cudaMalloc(&m_MemPtr, m_Size));
		}
	};
}

template<typename T>
using CudaMem = CudaAlloc::Memory<T>;

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
			<< file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}