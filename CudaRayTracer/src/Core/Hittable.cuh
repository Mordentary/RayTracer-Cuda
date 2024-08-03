#pragma once
#include "AABB.cuh"

namespace CRT
{
	class Material;
	class Lambertian;
	class Sphere;
	
	struct HitInfo {
		Vec3 Point;
		Vec3 Normal;
		int MaterialIndex;
		float IntersectionTime;
		bool IsNormalOutward;

		__device__ inline void setFaceNormal(const Ray& r, const Vec3& outward_normal) {
			IsNormalOutward = dot(r.getDirection(), outward_normal) < 0;
			Normal = IsNormalOutward ? outward_normal : -outward_normal;
		}
	};

	class Hittable
	{
	public:
		__device__ virtual bool hit(const Ray& ray, Interval ray_t, HitInfo& info) const = 0;
		__device__ AABB boundingBox() const { return m_BoundingBox; }

	protected:
		AABB m_BoundingBox;
	};
}