#pragma once

#include"Vec3.cuh"

namespace CRT
{
	class Texture {
	public:
		virtual ~Texture() = default;

		virtual Color value(double u, double v, const Point3& p) const = 0;
	};

	class SolidColor : public Texture {
	public:
		SolidColor(const Color& albedo) : m_Albedo(albedo) {}

		SolidColor(double red, double green, double blue) : SolidColor(Color(red, green, blue)) {}

		Color value(double u, double v, const Point3& p) const override 
		{
			return m_Albedo;
		}

	private:
		Color m_Albedo;
	};
}