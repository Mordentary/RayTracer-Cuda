#pragma once

#include "CUDAHelpers.h"
#include "CUDAKernels.h"
#include "Core/HittableList.cuh"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <set>

struct MeshData {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::vector<int> faceMaterialIds;
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

	// Device pointers
	int* m_dObjectsNum;
	CRT::HittableList* m_dWorld = nullptr;
	CRT::BVHNode* m_dBVHNodes = nullptr;
	CRT::Mesh* m_dMeshes = nullptr;
	Vertex* m_dVertices = nullptr;
	uint32_t* m_dIndices = nullptr;
	int* m_dFaceMaterialIds = nullptr;

	// Per-mesh offsets
	std::vector<uint32_t> m_VertexOffsets;
	std::vector<uint32_t> m_IndexOffsets;
	std::vector<uint32_t> m_VertexCounts;
	std::vector<uint32_t> m_IndexCounts;
	std::vector<uint32_t> m_FaceMatOffsets;
	std::vector<uint32_t> m_FaceCounts;     // how many faces in each mesh

	int m_NumMeshes;

	// Materials
	CRT::MaterialData* m_dMaterialsData = nullptr; // array of all MaterialData on device
	std::vector<CRT::MaterialData> m_SceneMaterialsData; // host side
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
	CUDA_CHECK(cudaFree(m_dFaceMaterialIds));
	cudaFree(m_dMaterialsData);
}

void SceneManager::initializeScene(const CUDAHelpers::RenderConfig& renderConfig, curandState* dRandState) {
	initMeshes(dRandState);

	if (!m_SceneMaterialsData.empty()) {
		cudaMalloc(&m_dMaterialsData, sizeof(CRT::MaterialData) * m_SceneMaterialsData.size());
		cudaMemcpy(m_dMaterialsData,
			m_SceneMaterialsData.data(),
			sizeof(CRT::MaterialData) * m_SceneMaterialsData.size(),
			cudaMemcpyHostToDevice);
	}

	CUDAKernels::createRandomWorld CUDA_KERNEL(1, 1) (m_dWorld, m_dMeshes, m_NumMeshes, m_dMaterialsData, (int)m_SceneMaterialsData.size(), m_dObjectsNum, dRandState);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	int maxNodes = 2 * (*m_dObjectsNum) - 1;
	CUDA_CHECK(cudaMalloc(&m_dBVHNodes, maxNodes * sizeof(CRT::BVHNode)));

	CUDAKernels::createBVH CUDA_KERNEL(1, 1) (m_dWorld, m_dBVHNodes, dRandState);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void SceneManager::initMeshes(curandState* dRandState) {
	std::vector<std::string> modelFiles = {
	"assets/models/CornellBox-Original.obj",
		"assets/models/bunny.obj" };
	std::vector<MeshData> allMeshData;
	for (const auto& file : modelFiles) {
		loadObject(file, allMeshData);
	}

	m_NumMeshes = (int)allMeshData.size();

	std::vector<Vertex>   allVertices;
	std::vector<uint32_t> allIndices;
	std::vector<int>      allFaceMatIds;

	// Reserve
	m_VertexOffsets.resize(m_NumMeshes);
	m_IndexOffsets.resize(m_NumMeshes);
	m_VertexCounts.resize(m_NumMeshes);
	m_IndexCounts.resize(m_NumMeshes);
	m_FaceMatOffsets.resize(m_NumMeshes);
	m_FaceCounts.resize(m_NumMeshes);
	std::vector<uint32_t> materialIDoffset;

	std::set<int> m_UniqueMaterialsPerMesh;
	for (int i = 0; i < m_NumMeshes; i++) {
		MeshData& md = allMeshData[i];

		uint32_t baseVertex = static_cast<uint32_t>(allVertices.size());
		uint32_t baseIndex = static_cast<uint32_t>(allIndices.size());
		uint32_t baseFaceMat = static_cast<uint32_t>(allFaceMatIds.size());

		m_VertexOffsets[i] = baseVertex;
		m_IndexOffsets[i] = baseIndex;
		m_FaceMatOffsets[i] = baseFaceMat;

		// Insert the mesh’s vertices
		allVertices.insert(allVertices.cend(), md.vertices.cbegin(), md.vertices.cend());
		allIndices.insert(allIndices.end(), md.indices.cbegin(), md.indices.cend());
		allFaceMatIds.insert(allFaceMatIds.cend(),
			md.faceMaterialIds.cbegin(),
			md.faceMaterialIds.cend());

		m_UniqueMaterialsPerMesh.clear();
		m_UniqueMaterialsPerMesh.insert(md.faceMaterialIds.cbegin(), md.faceMaterialIds.cend());
		materialIDoffset.push_back(m_UniqueMaterialsPerMesh.size());
		m_VertexCounts[i] = (uint32_t)md.vertices.size();
		m_IndexCounts[i] = (uint32_t)md.indices.size();
		m_FaceCounts[i] = (uint32_t)md.faceMaterialIds.size();
	}

	// Allocate GPU arrays
	cudaMalloc(&m_dVertices, allVertices.size() * sizeof(Vertex));
	cudaMalloc(&m_dIndices, allIndices.size() * sizeof(uint32_t));
	cudaMalloc(&m_dFaceMaterialIds, allFaceMatIds.size() * sizeof(int));

	cudaMemcpy(m_dVertices, allVertices.data(),
		allVertices.size() * sizeof(Vertex),
		cudaMemcpyHostToDevice);

	cudaMemcpy(m_dIndices, allIndices.data(),
		allIndices.size() * sizeof(uint32_t),
		cudaMemcpyHostToDevice);

	cudaMemcpy(m_dFaceMaterialIds, allFaceMatIds.data(),
		allFaceMatIds.size() * sizeof(int),
		cudaMemcpyHostToDevice);

	CUDA_CHECK(cudaMalloc(&m_dMeshes, m_NumMeshes * sizeof(CRT::Mesh)));
	// For each mesh, call initMesh
	for (int i = 0; i < m_NumMeshes; i++) {
		uint32_t vCount = m_VertexCounts[i];
		uint32_t iCount = m_IndexCounts[i];
		uint32_t vOffset = m_VertexOffsets[i];
		uint32_t iOffset = m_IndexOffsets[i];

		uint32_t faceMatOffset = m_FaceMatOffsets[i];
		uint32_t materialIDOffset = i == 0 ? 0 : materialIDoffset[i - 1];

		// Kernel call
		CUDAKernels::initMesh CUDA_KERNEL(1, 1) (
			m_dMeshes + i,
			m_dVertices,
			m_dIndices,
			m_dFaceMaterialIds,
			vCount,
			iCount,
			vOffset,
			iOffset,
			faceMatOffset,
			materialIDOffset,
			dRandState
			);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}
}

void SceneManager::loadObject(const std::string& filename, std::vector<MeshData>& meshDataList) {
	// 1. Load OBJ using TinyObj
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn, err;
	size_t last_slash_idx = filename.find_last_of("/\\");
	std::string base_dir;
	if (last_slash_idx != std::string::npos) {
		base_dir = filename.substr(0, last_slash_idx + 1); // Include the slash
	}
	else {
		base_dir = "./"; // Current directory if no path is found
	}

	// Load the OBJ file
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(), base_dir.c_str());
	if (!warn.empty()) std::cerr << "TinyObjLoader warning: " << warn << std::endl;
	if (!err.empty())  std::cerr << "TinyObjLoader error:   " << err << std::endl;
	if (!ret) throw std::runtime_error("Failed to load object: " + filename);

	size_t baseMatIndex = materials.size();
	// For each tinyobj material, build a MaterialData
	for (size_t i = 0; i < materials.size(); i++) {
		const auto& mat = materials[i];
		CRT::MaterialType mt = CRT::MaterialType::Lambertian;
		if (mat.emission[0] > 0.f || mat.emission[1] > 0.f || mat.emission[2] > 0.f) {
			mt = CRT::MaterialType::DiffuseLight;
		}
		else if (mat.dissolve < 1.f) {
			mt = CRT::MaterialType::Dielectric;
		}
		else if (mat.specular[0] > 0.f) {
			mt = CRT::MaterialType::Metal;
		}
		float r = 0.f;
		if (mt == CRT::MaterialType::Metal) {
			if (mat.roughness > 0.f) r = mat.roughness;
			else r = sqrtf(2.f / (mat.shininess + 2.f));
		}
		float ior = (mt == CRT::MaterialType::Dielectric ? mat.ior : 1.f);

		CRT::MaterialData mdata(mt,
			CRT::Vec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]),
			r,
			ior,
			CRT::Vec3(mat.emission[0], mat.emission[1], mat.emission[2]));
		m_SceneMaterialsData.push_back(mdata);
	}

	// For each shape, create one MeshData
	MeshData meshData;
	std::unordered_map<std::string, uint32_t> uniqueVerts;
	for (const auto& shape : shapes) {
		meshData.vertices.resize(attrib.vertices.size());

		size_t indexOffset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
			int faceVerts = shape.mesh.num_face_vertices[f]; // should be 3

			int faceMatId = 0;
			if (f < shape.mesh.material_ids.size()) {
				faceMatId = shape.mesh.material_ids[f];
				if (faceMatId < 0 || faceMatId >= static_cast<int>(m_SceneMaterialsData.size())) {
					faceMatId = 0;
				}
			}
			meshData.faceMaterialIds.push_back(faceMatId);

			for (int v = 0; v < faceVerts; v++) {
				tinyobj::index_t idx = shape.mesh.indices[indexOffset + v];

				Vertex vertex{};
				// Position
				vertex.Position = CRT::Vec3(
					attrib.vertices[3 * idx.vertex_index + 0],
					attrib.vertices[3 * idx.vertex_index + 1],
					attrib.vertices[3 * idx.vertex_index + 2]
				);

				// Normal
				if (idx.normal_index >= 0) {
					vertex.Normal = CRT::Vec3(
						attrib.normals[3 * idx.normal_index + 0],
						attrib.normals[3 * idx.normal_index + 1],
						attrib.normals[3 * idx.normal_index + 2]
					);
				}
				else {
					vertex.Normal = CRT::Vec3(0.f);
				}

				// UV
				if (idx.texcoord_index >= 0) {
					vertex.UV.e[0] = attrib.texcoords[2 * idx.texcoord_index + 0];
					vertex.UV.e[1] = attrib.texcoords[2 * idx.texcoord_index + 1];
				}
				else {
					vertex.UV = CRT::Vec2(0.f);
				}
				meshData.indices.push_back(idx.vertex_index);
				meshData.vertices[idx.vertex_index] = std::move(vertex);
			}
			indexOffset += faceVerts;
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
	float scale = 0.6f / (maxBounds - minBounds).maxComponent();

	for (auto& meshData : meshDataList) {
		for (auto& vertex : meshData.vertices) {
			vertex.Position = (vertex.Position - center) * scale;
		}
	}
	//std::cout << "Loaded " << meshDataList.size() << " shapes with "
	//	<< meshData.vertices.size() << " unique vertices and "
	//	<< meshData.indices.size() << " indices from " << filename << std::endl;
}
