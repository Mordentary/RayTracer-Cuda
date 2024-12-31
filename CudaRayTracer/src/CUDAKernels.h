#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "Core/Vec3.cuh"
#include "Core/Ray.cuh"
#include "Core/Camera.cuh"
#include "Core/BVHNode.cuh"
#include "Core/Sphere.cuh"
#include "Core/Utility.cuh"
#include "Core/Material.cuh"
#include "Core/Mesh.cuh"
#include "CRTUtility.cuh"

namespace CUDAKernels {
	__global__ void initRandState(curandState* rand_state, unsigned long long seed, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		int pixel_index = y * width + x;

		unsigned long long pixel_seed = seed + pixel_index;
		curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
	}

	__device__ void createMaterialsKernel(CRT::HittableList* world,
		const CRT::MaterialData* dMatData,
		int matCount)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			for (int i = 0; i < matCount; i++) {
				CRT::MaterialType t = dMatData[i].getType();
				CRT::Material* matPtr = nullptr;

				if (t == CRT::MaterialType::Lambertian) {
					matPtr = new CRT::Lambertian(dMatData[i].getAlbedo());
				}
				else if (t == CRT::MaterialType::Metal) {
					matPtr = new CRT::Metal(dMatData[i].getAlbedo(),
						dMatData[i].getRoughness());
				}
				else if (t == CRT::MaterialType::Dielectric) {
					matPtr = new CRT::Dielectric(dMatData[i].getIOR());
				}
				else if (t == CRT::MaterialType::DiffuseLight) {
					matPtr = new CRT::DiffuseLight(dMatData[i].getEmission());
				}

				world->addMaterial(matPtr);
			}
		}
	}

	__global__ void createRandomWorld(CRT::HittableList* world, CRT::Mesh* meshes, int numMeshes, CRT::MaterialData* materialsData, int materialsNum, int* objectsNum, curandState* rand_state) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			new (world) CRT::HittableList();

			//world->addMaterial(new CRT::DiffuseLight(CRT::Color(0.3, 0.1, 0.1)));

			createMaterialsKernel(world, materialsData, materialsNum);
			// Add all meshes to the world
			for (int i = 0; i < numMeshes; ++i)
			{
				world->add(&meshes[i]);
			}

			int ground_material_index = world->addMaterial(new CRT::Lambertian(CRT::Color(0.5, 0.5, 0.5)));
			world->add(new CRT::Sphere(CRT::Vec3(0, -1000, 0), 999, ground_material_index));

			int material3 = world->addMaterial(new CRT::Metal(CRT::Color(0.7, 0.6, 0.5), 0.0f));
			world->add(new CRT::Sphere(CRT::Vec3(0.3, 1, 0), 0.2f, material3));

			//for (int i = 0; i < world->s_MAX_MATERIALS; i++) {
			//	CRT::Color albedo = CRT::Utility::randomVector(0.2f, 1.0f, rand_state);
			//	float roughness = CRT::Utility::randomFloat(0.0f, 0.1f, rand_state);
			//	int materialIndex = world->addMaterial(new CRT::Lambertian(albedo));
			//}
			//world->add(meshes);

			*objectsNum = world->m_NumObjects;
		}
	}

	__global__ void createBVH(CRT::HittableList* objects, CRT::BVHNode* nodes, curandState* rand_state) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			CRT::BVHNode::buildBVHScene(objects, objects->m_NumObjects, nodes, rand_state);
		}
	}

	__global__ void initMesh(CRT::Mesh* mesh, Vertex* globalVertices, uint32_t* globalIndices, int* globalFaceMatIndices,
		uint32_t vertexCount, uint32_t indexCount,
		uint32_t vertexOffset, uint32_t indexOffset, uint32_t faceMatOffset, curandState* d_rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			new (mesh) CRT::Mesh(globalVertices, globalIndices, globalFaceMatIndices,
				vertexCount, indexCount, vertexOffset, indexOffset, faceMatOffset, d_rand_state);
		}
	}

	__device__ CRT::Color rayColor(const CRT::Ray& r, CRT::BVHNode* d_sceneRoot, CRT::HittableList* d_world, curandState* rand_state) {
		CRT::Ray current_ray = r;
		CRT::Color accumulated_color(1.0f, 1.0f, 1.0f);
		CRT::Color final_color(0.0f, 0.0f, 0.0f);
		const int MAX_BOUNCES = 10;
		const int MIN_BOUNCES = 3;
		const float MAX_PROB = 0.95f;

		for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
			CRT::HitInfo rec;

			if (bounce >= MIN_BOUNCES) {
				float survival_prob = fmaxf(accumulated_color.x(), fmaxf(accumulated_color.y(), accumulated_color.z()));
				survival_prob = fminf(survival_prob, MAX_PROB);

				if (curand_uniform(rand_state) > survival_prob) {
					break;  // Terminate the path
				}
				accumulated_color /= survival_prob;
			}

			if (d_sceneRoot->hit(current_ray, CRT::Interval(0.001f, INFINITY), rec)) {
				CRT::Ray scattered{};
				CRT::Color attenuation{};

				if (rec.MaterialIndex >= 0 && rec.MaterialIndex < d_world->m_NumMaterials) {
					if (d_world->m_Materials[rec.MaterialIndex]->scatter(current_ray, rec, attenuation, scattered, rand_state)) {
						accumulated_color *= attenuation;
						current_ray = scattered;
					}
					else {
						return d_world->m_Materials[rec.MaterialIndex]->emit();
					}
				}
			}
			else {
				// Ray didn't hit anything, add background color
				CRT::Color background_color = CRT::getBackgroundColor(current_ray);
				final_color = accumulated_color * background_color;
				break;
			}
		}
		return final_color;
	}

	__global__ void render(unsigned char* data, CRT::BVHNode* d_sceneRoot, CRT::HittableList* d_world, curandState* rand_state, int imageWidth, int imageHeight) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= imageWidth || y >= imageHeight) return;

		int pixel_index = y * imageWidth + x;
		curandState local_rand_state = rand_state[pixel_index];

		int data_pixel_index = pixel_index * 4;

		CRT::Color pixel_color(0, 0, 0);
		for (int sample = 0; sample < CRT::d_camera.m_SamplesPerPixel; sample++)
		{
			const CRT::Ray& r = CRT::d_camera.getRay(x, y, imageWidth, imageHeight, &local_rand_state);
			pixel_color += rayColor(r, d_sceneRoot, d_world, &local_rand_state);
		}

		CRT::writeColor(data, data_pixel_index, CRT::d_camera.m_PixelSampleScale * pixel_color);
		rand_state[pixel_index] = local_rand_state;
	}
}
