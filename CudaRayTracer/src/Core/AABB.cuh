#pragma once

#include"Interval.cuh"
#include"Ray.cuh"
#include "HitInfo.cuh"

namespace CRT
{
	struct AABB {
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

		__host__ __device__ Interval& axisInterval(int n)
		{
			if (n == 1) return y;
			if (n == 2) return z;
			return x;
		}

		__device__ void expand(const Point3& p)
		{
			x.min = fminf(x.min, p.x());
			x.max = fmaxf(x.max, p.x());
			y.min = fminf(y.min, p.y());
			y.max = fmaxf(y.max, p.y());
			z.min = fminf(z.min, p.z());
			z.max = fmaxf(z.max, p.z());
		}

		__device__ void expand(float range)
		{
			x = x.expand(range);
			y = y.expand(range);
			z = z.expand(range);
		}

		__device__ AABB expand(float range) const
		{
			return AABB(x.expand(range),
				y.expand(range),
				z.expand(range));
		}
		__device__ float area()
		{
			float ex = x.size();  
			float ey = y.size(); 
			float ez = z.size();

			return 2.0f * (ex * ey + ey * ez + ez * ex);
		}

		__device__ void expand(const AABB& other)
		{
			x = Interval(fminf(x.min, other.x.min), fmaxf(x.max, other.x.max));
			y = Interval(fminf(y.min, other.y.min), fmaxf(y.max, other.y.max));
			z = Interval(fminf(z.min, other.z.min), fmaxf(z.max, other.z.max));
			padToMinimums();
		}

		__device__ static AABB combine(const AABB& a, const AABB& b)
		{
			return AABB(
				Interval(fminf(a.x.min, b.x.min), fmaxf(a.x.max, b.x.max)),
				Interval(fminf(a.y.min, b.y.min), fmaxf(a.y.max, b.y.max)),
				Interval(fminf(a.z.min, b.z.min), fmaxf(a.z.max, b.z.max))
			);
		}

		//__device__ inline bool hit(const Ray& r, Interval ray_t) const
		//{
		//	const Vec3& origin = r.origin();
		//	const Vec3& dir = r.direction();
		//	const Vec3 invDir(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());

		//	float t1 = (x.min - origin.x()) * invDir.x();
		//	float t2 = (x.max - origin.x()) * invDir.x();
		//	float t3 = (y.min - origin.y()) * invDir.y();
		//	float t4 = (y.max - origin.y()) * invDir.y();
		//	float t5 = (z.min - origin.z()) * invDir.z();
		//	float t6 = (z.max - origin.z()) * invDir.z();

		//	float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
		//	float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

		//	// Check against the provided interval
		//	tmin = fmaxf(tmin, ray_t.min);
		//	tmax = fminf(tmax, ray_t.max);

		//	const float epsilon = 1e-8f;
		//	return tmax >= tmin + epsilon && tmax > 0;
		//}
		__device__ bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const
		{
			const Vec3& dir = r.direction();
			const Vec3 invD(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());
			Vec3 t0s = (min() - r.origin()) * invD;
			Vec3 t1s = (max() - r.origin()) * invD;

			float tmin = fmaxf(fmaxf(fminf(t0s.x(), t1s.x()), fminf(t0s.y(), t1s.y())), fminf(t0s.z(), t1s.z()));
			float tmax = fminf(fminf(fmaxf(t0s.x(), t1s.x()), fmaxf(t0s.y(), t1s.y())), fmaxf(t0s.z(), t1s.z()));

			tmin = fmaxf(tmin, ray_t.min);
			tmax = fminf(tmax, ray_t.max);

			if (tmax <= tmin)
				return false;

			rec.IntersectionTime = tmin;
			rec.Point = r.pointAtDistance(tmin);
			rec.Normal = Vec3(0.0f, 1.0f, 0.0f);  // Use a distinctive normal for AABBs
			rec.MaterialIndex = 0;  
			rec.IsNormalOutward = true;

			return true;
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
				(x.min + x.max) * 0.5f,
				(y.min + y.max) * 0.5f,
				(z.min + z.max) * 0.5f
			);
		}

		__device__ Point3 min() const
		{
			return Point3(x.min, y.min, z.min);
		}

		__device__ Point3 max() const
		{
			return Point3(x.max, y.max, z.max);
		}
	private:
		__device__ inline void swap(float& a, float& b) const
		{
			float temp = a;
			a = b;
			b = temp;
		}
		__device__ void padToMinimums() {
			float delta = 0.000001f;
			if (x.size() < delta) x = x.expand(delta);
			if (y.size() < delta) y = y.expand(delta);
			if (z.size() < delta) z = z.expand(delta);
		}
	};
#define AABB_EMPTY AABB(INTERVAL_EMPTY, INTERVAL_EMPTY, INTERVAL_EMPTY)
#define AABB_UNIVERSE AABB(INTERVAL_UNIVERSE, INTERVAL_UNIVERSE, INTERVAL_UNIVERSE)
}