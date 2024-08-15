#pragma once
#include"Utility.cuh"
namespace CRT
{
	struct Interval
	{
		float
			min = INFINITY_CRT,
			max = -INFINITY_CRT;

		__device__ __host__ Interval() = default;

		__device__ __host__ Interval(float Min, float Max) : min(Min), max(Max) {}

		__device__ __host__ Interval(const Interval& a, const Interval& b) {
			// Create the interval tightly enclosing the two input intervals.
			min = a.min <= b.min ? a.min : b.min;
			max = a.max >= b.max ? a.max : b.max;
		}

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
		__device__ Interval expand(float delta) const {
			float padding = delta / 2.f;
			return Interval(min - padding, max + padding);
		}

	};
	#define INTERVAL_EMPTY Interval()
	#define INTERVAL_UNIVERSE Interval(-INFINITY_CRT, +INFINITY_CRT)
}