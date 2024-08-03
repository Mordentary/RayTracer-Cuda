#pragma once
#include "hittable.cuh"

namespace CRT
{
	class Sphere : public Hittable
	{
	public:
		__device__  Sphere() = default;

		__device__ Sphere(const Vec3& center, float rad, int matIndex)
			: m_Center(center), m_Radius(rad), m_RadSquared(rad* rad), m_MaterialIndex(matIndex)
		{
			auto rvec = Vec3(rad, rad, rad);
			m_BoundingBox = AABB(center - rvec, center + rvec);
		}

		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			Vec3 oc = r.getOrigin() - m_Center;
			float a = dot(r.getDirection(), r.getDirection());
			float half_b = dot(oc, r.getDirection());
			float c = dot(oc, oc) - m_RadSquared;
			float discriminant = half_b * half_b - a * c;

			if (discriminant < 0)
				return false;

			float sqrtd = sqrt(discriminant);

			// Find the nearest root that lies in the acceptable range.
			float root = (-half_b - sqrtd) / a;

			if (ray_t.outOfInterval(root)) {
				root = (-half_b + sqrtd) / a;
				if (ray_t.outOfInterval(root))
					return false;
			}

			rec.IntersectionTime = root;
			rec.Point = r.pointAtDistance(rec.IntersectionTime);
			rec.MaterialIndex = m_MaterialIndex;
			Vec3 outward_normal = (rec.Point - m_Center) / m_Radius;
			rec.setFaceNormal(r, outward_normal);

			return true;
		}

	private:
		Vec3 m_Center;
		float m_Radius, m_RadSquared;
		 int m_MaterialIndex;
	};
}