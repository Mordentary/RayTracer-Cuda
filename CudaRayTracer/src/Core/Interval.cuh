#pragma once
#include"Utility.cuh"
namespace CRT
{
	struct Interval
	{
		float
			Min = INFINITY_CRT,
			Max = -INFINITY_CRT;

		__device__ __host__ Interval() = default;

		__device__ __host__ Interval(float Min, float Max) : Min(Min), Max(Max) {}

		__device__ __host__ Interval(const Interval& a, const Interval& b) {
			// Create the interval tightly enclosing the two input intervals.
			Min = a.Min <= b.Min ? a.Min : b.Min;
			Max = a.Max >= b.Max ? a.Max : b.Max;
		}

		__device__ float size() const {
			return Max - Min;
		}

		__device__ bool contains(float x) const {
			return Min <= x && x <= Max;
		}

		__device__ bool surrounds(float x) const {
			return Min < x && x < Max;
		}

		__device__ bool outOfInterval(float x) const {
			return x < Min || x > Max;
		}

		__device__ float clamp(float x) const {
			if (x < Min) return Min;
			if (x > Max) return Max;
			return x;
		}
		__device__ Interval expand(float delta) const {
			float padding = delta / 2.f;
			return Interval(Min - padding, Max + padding);
		}

	};
	#define INTERVAL_EMPTY Interval(+INFINITY_CRT, -INFINITY_CRT)
	#define INTERVAL_UNIVERSE Interval(-INFINITY_CRT, +INFINITY_CRT)
}