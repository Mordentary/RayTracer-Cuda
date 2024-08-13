#pragma once

#include"Interval.cuh"
#include"Ray.cuh"

namespace CRT
{
	class AABB {
	public:
		Interval x, y, z;

		__device__ AABB() {}

		__device__ AABB(const Interval& x, const Interval& y, const Interval& z)
			: x(x), y(y), z(z)
		{
			padToMinimums();
		}

		__device__ AABB(const AABB& box0, const AABB& box1)
		{
			x = Interval(box0.x, box1.x);
			y = Interval(box0.y, box1.y);
			z = Interval(box0.z, box1.z);
		}

		__device__ 	AABB(const Point3& a, const Point3& b)
		{
			x = Interval(std::fmin(a[0], b[0]), std::fmax(a[0], b[0]));
			y = Interval(std::fmin(a[1], b[1]), std::fmax(a[1], b[1]));
			z = Interval(std::fmin(a[2], b[2]), std::fmax(a[2], b[2]));

			padToMinimums();
		}

		__host__ __device__ const Interval& axisInterval(int n) const
		{
			if (n == 1) return y;
			if (n == 2) return z;
			return x;
		}

		__device__ inline bool hit(const Ray& r, Interval ray_t) const
		{
			const Vec3& origin = r.origin();
			const Vec3& dir = r.direction();
			const Vec3 invDir(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());

			float t1 = (x.Min - origin.x()) * invDir.x();
			float t2 = (x.Max - origin.x()) * invDir.x();
			float t3 = (y.Min - origin.y()) * invDir.y();
			float t4 = (y.Max - origin.y()) * invDir.y();
			float t5 = (z.Min - origin.z()) * invDir.z();
			float t6 = (z.Max - origin.z()) * invDir.z();

			float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
			float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

			return tmax >= tmin && tmax > 0;
		}

		__device__ int longestAxis() const
		{
			if (x.size() > y.size())
				return x.size() > z.size() ? 0 : 2;
			else
				return y.size() > z.size() ? 1 : 2;
		}

		__device__ Point3 center() const
		{
			return Point3(
				(x.Min + x.Max) * 0.5f,
				(y.Min + y.Max) * 0.5f,
				(z.Min + z.Max) * 0.5f
			);
		}


	private:
		__device__ void padToMinimums() {
			double delta = 0.0001;
			if (x.size() < delta) x = x.expand(delta);
			if (y.size() < delta) y = y.expand(delta);
			if (z.size() < delta) z = z.expand(delta);
		}

	};
#define AABB_EMPTY AABB(INTERVAL_EMPTY, INTERVAL_EMPTY, INTERVAL_EMPTY)
#define AABB_UNIVERSE AABB(INTERVAL_UNIVERSE, INTERVAL_UNIVERSE, INTERVAL_UNIVERSE)
}