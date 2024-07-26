#pragma once
#include "Core.cuh"
#include "Vec3.cuh"  

namespace CRT
{
	class Ray
	{
	public:
		__device__ Ray() = default;
		__device__ Ray(const Vec3& origin, const Vec3& direction)
			: m_Origin(origin), m_Direction(direction) {}

		__device__ inline Vec3 pointAtDistance(float distance) const
		{
			return m_Origin + distance * m_Direction;
		}
		__device__ inline const Vec3& getOrigin() const { return m_Origin; }
		__device__ inline const Vec3& getDirection() const { return m_Direction; }

	private:
		Vec3 m_Origin{};
		Vec3 m_Direction{};
	};


}
