#pragma once
#include"AABB.cuh"
#include "HitInfo.cuh"

namespace CRT
{
	class Material;
	class Lambertian;
	class Sphere;


	class Hittable
	{
	public:
		__device__ virtual inline bool hit(const Ray& ray, Interval ray_t, HitInfo& info) const = 0;
		__device__ AABB boundingBox() const { return m_BoundingBox; }

	protected:
		AABB m_BoundingBox;
	};
}