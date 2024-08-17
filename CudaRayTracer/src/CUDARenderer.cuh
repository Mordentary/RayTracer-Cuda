#pragma once

#include "CUDAHelpers.h"
#include "Core/BVHNode.cuh"
#include "Core/HittableList.cuh"
#include "Core/Camera.cuh"
#include "CUDAKernels.h"

class CUDARenderer {
public:
	CUDARenderer(int width, int height);
	~CUDARenderer();

	void initialize(const CUDAHelpers::RenderConfig& config);
	void updateCamera(const CRT::Camera& camera);
	void render(CRT::BVHNode* bvhNodes, CRT::HittableList* world);
	unsigned char* getImageData() const { return m_dImageData; }
	curandState* getRandState() const { return m_dRandState; }

private:
	int m_Width;
	int m_Height;
	CUDAHelpers::RenderConfig m_RenderConfig;

	unsigned char* m_dImageData;
	curandState* m_dRandState;
};



CUDARenderer::CUDARenderer(int width, int height)
	: m_Width(width), m_Height(height), m_dImageData(nullptr), m_dRandState(nullptr) {}

CUDARenderer::~CUDARenderer() {
	if (m_dImageData) cudaFree(m_dImageData);
	if (m_dRandState) cudaFree(m_dRandState);
}

void CUDARenderer::initialize(const CUDAHelpers::RenderConfig& config) {
	m_RenderConfig = config;

	CUDA_CHECK(cudaMalloc(&m_dImageData, m_Width * m_Height * 4 * sizeof(unsigned char)));
	CUDA_CHECK(cudaMalloc(&m_dRandState, m_RenderConfig.totalThreads * sizeof(curandState)));

	// Initialize random states
	CUDAKernels::initRandState CUDA_KERNEL(m_RenderConfig.blocks, m_RenderConfig.threads) (m_dRandState, rand(), m_Width, m_Height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void CUDARenderer::updateCamera(const CRT::Camera& camera) {
	CUDA_CHECK(cudaMemcpyToSymbol(CRT::d_camera, &camera, sizeof(CRT::Camera)));
}

void CUDARenderer::render(CRT::BVHNode* bvhNodes, CRT::HittableList* world) {
	CUDAKernels::render CUDA_KERNEL(m_RenderConfig.blocks, m_RenderConfig.threads) (
		m_dImageData, bvhNodes, world, m_dRandState, m_Width, m_Height);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}