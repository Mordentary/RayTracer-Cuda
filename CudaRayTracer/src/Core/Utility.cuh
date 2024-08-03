#pragma once
#include <curand_kernel.h>
#include"vec3.cuh"

namespace CRT
{
	//Constants
	__device__ static constexpr float INFINITY_CRT = std::numeric_limits<float>::infinity();
	__device__ static constexpr float PI = 3.1415926535897932385f;

	class Utility
	{
	public:

		//VectorFunc

		//RandomFunc
		__device__ static float randomFloat(float min, float max, curandState* local_rand_state)
		{
			return min + (max - min) * curand_uniform(local_rand_state);
		}

		__device__ static float randomFloat(curandState* local_rand_state)
		{
			return curand_uniform(local_rand_state);
		}
		__device__ static int randomInt(int min, int max, curandState* local_rand_state)
		{
			float random_float = randomFloat(static_cast<float>(min), static_cast<float>(max) + 1.0f, local_rand_state);
			return static_cast<int>(random_float);
		}

		__device__ static Vec3 randomVector(curandState* local_rand_state)
		{
			return Vec3(randomFloat(local_rand_state), randomFloat(local_rand_state), randomFloat(local_rand_state));
		}

		__device__ static Vec3 randomVector(float min, float max, curandState* local_rand_state)
		{
			return Vec3(randomFloat(min, max, local_rand_state),
				randomFloat(min, max, local_rand_state),
				randomFloat(min, max, local_rand_state));
		}

		__device__ static Vec3 randomPointInUnitSphere(curandState* local_rand_state)
		{
			while (true)
			{
				Vec3 point = randomVector(-1, 1, local_rand_state);
				if ((point).lengthSquared() >= 1) continue;
				return point;
			}
		}

		__device__ static Vec3 randomPointInUnitDisk(curandState* local_rand_state)
		{
			while (true) {
				Vec3 point = Vec3(randomFloat(-1, 1, local_rand_state), randomFloat(-1, 1, local_rand_state), 0);
				if ((point).lengthSquared() >= 1) continue;
				return point;
			}
		}

		__device__ static Vec3 randomPointInHemisphere(const Vec3& normal, curandState* local_rand_state)
		{
			Vec3 point = randomPointInUnitSphere(local_rand_state);
			if (dot(point, normal) > 0.0f) // In the same hemisphere as the normal
				return point;
			else
				return -point;
		}

		__device__ static Vec3 randomUnitVector(curandState* local_rand_state)
		{
			return unitVector(randomPointInUnitSphere(local_rand_state));
		}

		__device__ static Vec3 refract(const Vec3& incidentRay, const Vec3& normal, float eta)
		{
			float cos_theta = fminf(dot(-incidentRay, normal), 1.0f);
			Vec3 perp = eta * (incidentRay + cos_theta * normal);
			Vec3 parallel = -sqrtf(fabsf(1.0f - (perp).lengthSquared())) * normal;
			return perp + parallel;
		}

		__device__ static Vec3 randomVec3(curandState* rand_state) {
			return Vec3(Utility::randomFloat(rand_state), Utility::randomFloat(rand_state), Utility::randomFloat(rand_state));
		}

		__device__ static Vec3 randomVec3(float min, float max, curandState* rand_state) {
			return Vec3(Utility::randomFloat(min, max, rand_state), Utility::randomFloat(min, max, rand_state), Utility::randomFloat(min, max, rand_state));
		}

		__device__ static inline Vec3 randomInUnitSphere(curandState* rand_state) {
			while (true) {
				auto p = randomVec3(-1, 1, rand_state);
				if (p.lengthSquared() < 1)
					return p;
			}
		}
		__device__ static inline Vec3 randomOnHemisphere(const Vec3& normal, curandState* rand_state) {
			Vec3 on_unit_sphere = randomUnitVector(rand_state);
			if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
				return on_unit_sphere;
			else
				return -on_unit_sphere;
		}
	};
}