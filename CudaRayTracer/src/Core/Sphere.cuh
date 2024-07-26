#pragma once
#include "hittable.cuh"

namespace CRT
{
	class Sphere : public Hittable
	{
	public:
		__device__  Sphere() = default;

		__device__ Sphere(const Vec3& center, float rad, Material* mat)
			: m_Center(center), m_Radius(rad), m_RadiusSquared(rad* rad), m_Material(mat)
		{
		}

		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			Vec3 oc = r.getOrigin() - m_Center;
			float a = dot(r.getDirection(), r.getDirection());
			float half_b = dot(oc, r.getDirection());
			float c = dot(oc, oc) - m_RadiusSquared;
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
			rec.MaterialPtr = m_Material;
			Vec3 outward_normal = (rec.Point - m_Center) / m_Radius;
			rec.setFaceNormal(r, outward_normal);


			return true;
		}

	private:
		Vec3 m_Center;
		float m_Radius, m_RadiusSquared;
		Material* m_Material;

	};
}