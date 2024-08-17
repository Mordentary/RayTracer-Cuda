#pragma once

#include "CUDAHelpers.h"
#include "CUDAKernels.h"
#include "Core/HittableList.cuh"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include <unordered_map>

struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::string materialName;
};

class SceneManager {
public:
	SceneManager(int width, int height);
	~SceneManager();

	void initializeScene(const CUDAHelpers::RenderConfig& renderConfig, curandState* dRandState);
	CRT::BVHNode* getBVHNodes() const { return m_dBVHNodes; }
	CRT::HittableList* getWorld() const { return m_dWorld; }

private:
	void initMeshes(curandState* dRandState);
	void loadObject(const std::string& filename, std::vector<MeshData>& meshDataList);

	int m_Width;
	int m_Height;
	int* m_dObjectsNum;
	CRT::HittableList* m_dWorld = nullptr;
	CRT::BVHNode* m_dBVHNodes = nullptr;

	CRT::Mesh* m_dMeshes = nullptr;
	Vertex* m_dVertices = nullptr;
	uint32_t* m_dIndices = nullptr;

	// New members for multiple meshes and materials
	std::vector<uint32_t> m_VertexOffsets;
	std::vector<uint32_t> m_IndexOffsets;
	std::vector<uint32_t> m_VertexCounts;
	std::vector<uint32_t> m_IndexCounts;
	//std::vector<int> m_MaterialIndices;
	int m_NumMeshes;

	//// Material management
	//std::unordered_map<std::string, int> m_MaterialMap;
	//std::vector<CRT::Material*> m_Materials;
};

// Implementation

SceneManager::SceneManager(int width, int height) : m_Width(width), m_Height(height), m_NumMeshes(0) {
	CUDA_CHECK(cudaMalloc(&m_dWorld, sizeof(CRT::HittableList)));
	CUDA_CHECK(cudaMallocManaged(&m_dObjectsNum, sizeof(int)));
}

SceneManager::~SceneManager() {
	CUDA_CHECK(cudaFree(m_dWorld));
	CUDA_CHECK(cudaFree(m_dObjectsNum));
	CUDA_CHECK(cudaFree(m_dBVHNodes));
	CUDA_CHECK(cudaFree(m_dMeshes));
	CUDA_CHECK(cudaFree(m_dVertices));
	CUDA_CHECK(cudaFree(m_dIndices));

	//for (auto material : m_Materials) {
	//	delete material;
	//}
}

void SceneManager::initializeScene(const CUDAHelpers::RenderConfig& renderConfig, curandState* dRandState) {
	initMeshes(dRandState);

	CUDAKernels::createRandomWorld CUDA_KERNEL(1, 1) (m_dWorld, m_dMeshes, m_NumMeshes, nullptr, m_dObjectsNum, dRandState);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	int maxNodes = 2 * (*m_dObjectsNum) - 1;
	CUDA_CHECK(cudaMalloc(&m_dBVHNodes, maxNodes * sizeof(CRT::BVHNode)));

	CUDAKernels::createBVH CUDA_KERNEL(1, 1) (m_dWorld, m_dBVHNodes, dRandState);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void SceneManager::initMeshes(curandState* dRandState) {
	std::vector<std::string> modelFiles = { "assets/models/bunny.obj" , "assets/models/cornell-box.obj" };
		std::vector<MeshData> allMeshData;

	for (const auto& file : modelFiles)
	{
		loadObject(file, allMeshData);
	}

	m_NumMeshes = allMeshData.size();
	std::vector<Vertex> allVertices;
	std::vector<uint32_t> allIndices;

	for (const auto& meshData : allMeshData) {
		m_VertexOffsets.push_back(allVertices.size());
		m_IndexOffsets.push_back(allIndices.size());
		m_VertexCounts.push_back(meshData.vertices.size());
		m_IndexCounts.push_back(meshData.indices.size());

		allVertices.insert(allVertices.end(), meshData.vertices.begin(), meshData.vertices.end());
		allIndices.insert(allIndices.end(), meshData.indices.begin(), meshData.indices.end());
	}

	CUDA_CHECK(cudaMalloc(&m_dVertices, allVertices.size() * sizeof(Vertex)));
	CUDA_CHECK(cudaMalloc(&m_dIndices, allIndices.size() * sizeof(uint32_t)));
	CUDA_CHECK(cudaMalloc(&m_dMeshes, m_NumMeshes * sizeof(CRT::Mesh)));

	CUDA_CHECK(cudaMemcpy(m_dVertices, allVertices.data(), allVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(m_dIndices, allIndices.data(), allIndices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

	for (int i = 0; i < m_NumMeshes; ++i)
	{
		CUDAKernels::initMesh CUDA_KERNEL(1, 1) (m_dMeshes + i, m_dVertices, m_dIndices,
			m_VertexCounts[i], m_IndexCounts[i], m_VertexOffsets[i], m_IndexOffsets[i], 0, dRandState);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}

void SceneManager::loadObject(const std::string& filename, std::vector<MeshData>& meshDataList) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());

	materials.data()->

	if (!err.empty()) {
		std::cerr << "TinyObjLoader error: " << err << std::endl;
	}

	if (!ret) {
		throw std::runtime_error("Failed to load object file");
	}

	std::unordered_map<std::string, uint32_t> uniqueVertices;

	MeshData meshData;
	// Process each shape in the model
	for (const auto& shape : shapes) {
		// Process all vertices and indices for this shape
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex;
			vertex.Position = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			if (index.normal_index >= 0) {
				vertex.Normal = {
					attrib.normals[3 * index.normal_index + 0],
					attrib.normals[3 * index.normal_index + 1],
					attrib.normals[3 * index.normal_index + 2]
				};
			}

			if (index.texcoord_index >= 0) {
				vertex.UV = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					attrib.texcoords[2 * index.texcoord_index + 1]
				};
			}

			// Create a unique string key for the vertex
			std::string key = std::to_string(vertex.Position.x()) + "," +
				std::to_string(vertex.Position.y()) + "," +
				std::to_string(vertex.Position.z()) + "," +
				std::to_string(vertex.Normal.x()) + "," +
				std::to_string(vertex.Normal.y()) + "," +
				std::to_string(vertex.Normal.z()) + "," +
				std::to_string(vertex.UV.x()) + "," +
				std::to_string(vertex.UV.y());

			if (uniqueVertices.count(key) == 0) {
				// If this is a new unique vertex, add it to the list
				uniqueVertices[key] = static_cast<uint32_t>(meshData.vertices.size());
				meshData.vertices.push_back(vertex);
			}

			// Add the index
			meshData.indices.push_back(uniqueVertices[key]);
		}
	}
	meshDataList.push_back(meshData);


	// Calculate bounding box and normalize
	CRT::Vec3 minBounds(std::numeric_limits<float>::max());
	CRT::Vec3 maxBounds(std::numeric_limits<float>::lowest());

	for (auto& meshData : meshDataList) {
		for (const auto& vertex : meshData.vertices) {
			minBounds = CRT::Vec3::min(minBounds, vertex.Position);
			maxBounds = CRT::Vec3::max(maxBounds, vertex.Position);
		}
	}

	CRT::Vec3 center = (minBounds + maxBounds) * 0.5f;
	float scale = 2.0f / (maxBounds - minBounds).maxComponent();

	for (auto& meshData : meshDataList) {
		for (auto& vertex : meshData.vertices) {
			vertex.Position = (vertex.Position - center) * scale;
		}
	}

	std::cout << "Loaded " << meshDataList.size() << " shapes with "
		<< meshData.vertices.size() << " unique vertices and "
		<< meshData.indices.size() << " indices from " << filename << std::endl;
}
