#pragma once
#include"Hittable.h"

namespace BRT
{
	class Sphere : public Hittable
	{

	public:
		Sphere() = default;
		Sphere(const glm::vec3& center, double rad, const Shared<Material>& mat) : m_Center(center), m_Material(mat), m_Radius(rad), m_RadiusSquared(rad* rad) {};

		virtual bool Hit(const Ray& ray, double tMin, double tMax, HitInfo& info) const override;


	private:
		glm::vec3 m_Center;
		Shared<Material> m_Material;
		double m_Radius, m_RadiusSquared;
	};
}

