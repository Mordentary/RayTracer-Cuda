#pragma once

#include "Core/Vec3.cuh"
#include "Core/Ray.cuh"
#include "Core/Interval.cuh"

namespace CRT
{
	__device__ inline Vec3 linearToGamma(const Color& linearColor)
	{
		return Vec3(sqrt(linearColor.x()), sqrt(linearColor.y()), sqrt(linearColor.z()));
	}

	__device__ inline double linearToGamma(double component)
	{
		if (component > 0)
			return sqrt(component);
		return 0;
	}

	__device__ inline void writeColor(unsigned char* data, int index, const Vec3& color)
	{
		float r = linearToGamma(color.x());
		float g = linearToGamma(color.y());
		float b = linearToGamma(color.z());

		Interval defaultIntensity(0.000f, 0.999f);
		data[index] = static_cast<unsigned char>(256 * defaultIntensity.clamp(r));
		data[index + 1] = static_cast<unsigned char>(256 * defaultIntensity.clamp(g));
		data[index + 2] = static_cast<unsigned char>(256 * defaultIntensity.clamp(b));
		data[index + 3] = 255;
	}

	__device__ inline Color getBackgroundColor(const Ray& r) {
		Vec3 unit_direction = unitVector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
	}
}