#pragma once
#include"Utility.cuh"
namespace CRT
{


	class Interval
	{
	public:
		float min = -INFINITY_CRT, max = INFINITY_CRT;

		__device__ __host__ Interval() = default;

		__device__ __host__ Interval(float min, float max) : min(min), max(max) {}


		__device__ float size() const {
			return max - min;
		}

		__device__ bool contains(float x) const {
			return min <= x && x <= max;
		}

		__device__ bool surrounds(float x) const {
			return min < x && x < max;
		}

		__device__ bool outOfInterval(float x) const {
			return x < min || x > max;
		}

		__device__ float clamp(float x) const {
			if (x < min) return min;
			if (x > max) return max;
			return x;
		}
	};


}