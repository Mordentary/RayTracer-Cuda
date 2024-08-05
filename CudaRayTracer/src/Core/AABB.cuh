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
			: x(x), y(y), z(z) {}

		__device__ AABB(const AABB& box0, const AABB& box1) {
			x = Interval(box0.x, box1.x);
			y = Interval(box0.y, box1.y);
			z = Interval(box0.z, box1.z);
		}

		__device__ 	AABB(const Point3& a, const Point3& b) {
			x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
			y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
			z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
		}

		__host__ __device__ const Interval& axisInterval(int n) const
		{
			if (n == 1) return y;
			if (n == 2) return z;
			return x;
		}

		__device__ bool hit(const Ray& r, Interval ray_t) const {
			//const Point3& ray_orig = r.origin();
			//const Vec3& ray_dir = r.direction();

			//for (int axis = 0; axis < 3; axis++) {
			//	const Interval& ax = axisInterval(axis);
			//	const double adinv = 1.0 / ray_dir[axis];

			//	auto t0 = (ax.Min - ray_orig[axis]) * adinv;
			//	auto t1 = (ax.Max - ray_orig[axis]) * adinv;

			//	if (t0 < t1) {
			//		if (t0 > ray_t.Min) ray_t.Min = t0;
			//		if (t1 < ray_t.Max) ray_t.Max = t1;
			//	}
			//	else {
			//		if (t1 > ray_t.Min) ray_t.Min = t1;
			//		if (t0 < ray_t.Max) ray_t.Max = t0;
			//	}

			//	if (ray_t.Max <= ray_t.Min)
			//		return false;
			//}
			//return true;

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

		__device__ int longestAxis() const {
			if (x.size() > y.size())
				return x.size() > z.size() ? 0 : 2;
			else
				return y.size() > z.size() ? 1 : 2;
		}

		__device__ Point3 center() const {
			return Point3(
				(x.Min + x.Max) * 0.5f,
				(y.Min + y.Max) * 0.5f,
				(z.Min + z.Max) * 0.5f
			);
		}
	};
#define AABB_EMPTY AABB(INTERVAL_EMPTY, INTERVAL_EMPTY, INTERVAL_EMPTY)
#define AABB_UNIVERSE AABB(INTERVAL_UNIVERSE, INTERVAL_UNIVERSE, INTERVAL_UNIVERSE)
}