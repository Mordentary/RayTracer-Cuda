#pragma once 
#include "Ray.cuh"


namespace CRT
{
	struct HitInfo {
		Vec3 Point;
		Vec3 Normal;
		uint32_t MaterialIndex;
		float IntersectionTime;
		bool IsNormalOutward;
		float U_TexCoord, V_TexCoord;

		__device__ inline void setFaceNormal(const Ray& r, const Vec3& outward_normal) {
			IsNormalOutward = dot(r.direction(), outward_normal) < 0;
			Normal = IsNormalOutward ? outward_normal : -outward_normal;
		}
	};
}