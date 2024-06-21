#include "Sphere.h"
#include <glm/gtx/norm.hpp>


namespace BRT
{
	bool Sphere::Hit(const Ray& ray, double tMin, double tMax, HitInfo& info) const
	{
		glm::vec3 oc = ray.GetOrigin() - m_Center;
		double a = glm::dot(ray.GetDirection(), ray.GetDirection());
		double half_b = glm::dot(oc, ray.GetDirection());
		double c = glm::dot(oc, oc) - m_RadiusSquared;
		double discriminant = half_b * half_b - a * c;


		if (discriminant < 0.0) 
			return false;


		double sqrtd = 0.0;

		if (discriminant > 0) 
			sqrtd = sqrt(discriminant);
		

		// Find the nearest root that lies in the acceptable range.
		double root = (-half_b - sqrtd) / a;
		if (root < tMin || tMax < root) {
			root = (-half_b + sqrtd) / a;
			if (root < tMin || tMax < root)
				return false;
		}

		info.IntersectionTime = root;
		info.Point = ray.PointAtDistance(root);
		info.Normal = (info.Point - m_Center) / (float)m_Radius;
		info.SetFaceNormal(ray, info.Normal);
		info.Material = m_Material;

		return true;
	}


}