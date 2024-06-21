#pragma once
#include "Core/Core.h"





namespace BRT
{
	class HitInfo;
	class Ray;

	class Material
	{
	public:
		virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const = 0;

	};

	class Matte : public Material
	{
	public: 
		Matte(const glm::vec3& color) : m_Albedo(color) {}
		virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const override;
	private: 
		glm::vec3 m_Albedo;
	};

	class Metal : public Material
	{
	public:
		Metal(const glm::vec3& color, float roughness) : m_Albedo(color), m_Roughness(roughness) {}
		virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const override;
	private:
		glm::vec3 m_Albedo;
		float m_Roughness;
	};

	class Dialectric : public Material
	{
	public:
		Dialectric(float ior) : m_IndexOfRefraction(ior){}
		virtual bool Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const override;
	private:
		float m_IndexOfRefraction;
	private:
		double SchlicksReflectance(double cosine, double ref_idx) const;
	};


}

