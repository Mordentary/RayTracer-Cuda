#pragma once
#include "Hittable.cuh"
#include "Ray.cuh"
#include "Vec3.cuh"

namespace CRT
{
	class Material {
	public:
		__device__ virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const = 0;
		__device__ virtual ~Material() {};
	};

	class Lambertian : public Material
	{
	public:
		__device__ Lambertian(const Vec3& color) : m_Albedo(color)
		{
		}
		__device__ virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const override
		{
			Vec3 scatter_direction = rec.Normal + Utility::randomUnitVector(rand_state);

			if (scatter_direction.nearZero())
				scatter_direction = rec.Normal;

			// Update the ray for the next iteration
			scattered = Ray(rec.Point, scatter_direction);
			attenuation = m_Albedo;
			return true;
		}

	private:
		Vec3 m_Albedo;
	};

	class Metal : public Material
	{
	public:
		__device__ Metal(const Vec3& color, float roughness) : m_Albedo(color), m_Roughness(roughness < 1.f ? roughness : 1.f)
		{
		}
		__device__ virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const override
		{
			Vec3 reflected = reflect(rayIn.direction(), rec.Normal);
			reflected = unitVector(reflected) + (m_Roughness * Utility::randomUnitVector(rand_state));
			scattered = Ray(rec.Point, reflected);
			attenuation = m_Albedo;
			return (dot(scattered.direction(), rec.Normal) > 0);
		}

	private:
		Vec3 m_Albedo;
		float m_Roughness;
	};

	class Dielectric : public Material
	{
	public:
		__device__ Dielectric(float ior) : m_IndexOfRefraction(ior)
		{
		}
		__device__ virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, Color& attenuation, Ray& scattered, curandState* rand_state) const override
		{
			attenuation = Color(1.0, 1.0, 1.0);
			float ri = rec.IsNormalOutward ? (1.0f / m_IndexOfRefraction) : m_IndexOfRefraction;

			Vec3 unitDirection = unitVector(rayIn.direction());
			double cos_theta = fminf(dot(-unitDirection, rec.Normal), 1.0);
			double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

			bool cannot_refract = ri * sin_theta > 1.0;
			Vec3 dirR_out;
			//if (cannot_refract || schlicksReflectance(cos_theta, ri) > Utility::randomFloat(rand_state))
			if (cannot_refract || schlicksReflectance(cos_theta, ri) > Utility::randomFloat(rand_state))
				dirR_out = reflect(unitDirection, rec.Normal);
			else
				dirR_out = refract(unitDirection, rec.Normal, ri);

			scattered = Ray(rec.Point, dirR_out);
			return true;
		}

	private:
		float m_IndexOfRefraction = 1.0f;
		__device__ float schlicksReflectance(float cosine, float ref_idx) const
		{
			float r0 = (1 - ref_idx) / (1 + ref_idx);
			r0 = r0 * r0;
			return r0 + (1 - r0) * pow((1 - cosine), 5);
		}
	};
}