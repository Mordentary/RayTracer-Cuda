#pragma once
#include <cmath>
#include <iostream>
#include "cuda_runtime.h"

namespace CRT
{
	class Vec3
	{
	public:
		float e[3];

		__host__ __device__ Vec3() {};
		__host__ __device__ Vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}
		__host__ __device__ Vec3(float e) : e{ e, e, e } {}

		__host__ __device__ float x() const { return e[0]; }
		__host__ __device__ float y() const { return e[1]; }
		__host__ __device__ float z() const { return e[2]; }

		__host__ __device__ Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
		__host__ __device__ float operator[](int i) const { return e[i]; }
		__host__ __device__ float& operator[](int i) { return e[i]; }

		__host__ __device__ bool operator==(const Vec3& v) const {
			return e[0] == v.e[0] && e[1] == v.e[1] && e[2] == v.e[2];
		}

		__host__ __device__ bool operator!=(const Vec3& v) const {
			return !(*this == v);
		}

		__host__ __device__ Vec3& operator+=(const Vec3& v) {
			e[0] += v.e[0];
			e[1] += v.e[1];
			e[2] += v.e[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(const Vec3& v) {
			e[0] *= v.e[0];
			e[1] *= v.e[1];
			e[2] *= v.e[2];
			return *this;
		}

		__host__ __device__ Vec3& operator-=(const Vec3& v) {
			e[0] -= v.e[0];
			e[1] -= v.e[1];
			e[2] -= v.e[2];
			return *this;
		}

		__host__ __device__ Vec3& operator*=(float t) {
			e[0] *= t;
			e[1] *= t;
			e[2] *= t;
			return *this;
		}

		__host__ __device__ Vec3& operator/=(float t) {
			return *this *= 1 / t;
		}
		__host__ __device__ float maxComponent() const {
			return fmaxf(e[0], fmaxf(e[1], e[2]));
		}
		__host__ __device__ static Vec3 min(const Vec3& a, const Vec3& b)
		{
			return Vec3(fminf(a.x(), b.x()),
				fminf(a.y(), b.y()),
				fminf(a.z(), b.z()));
		}

		__host__ __device__ static Vec3 max(const Vec3& a, const Vec3& b)
		{
			return Vec3(fmaxf(a.x(), b.x()),
				fmaxf(a.y(), b.y()),
				fmaxf(a.z(), b.z()));
		}

		// Operator overloads for component-wise min and max
		__host__ __device__ friend Vec3 min(const Vec3& a, const Vec3& b)
		{
			return Vec3::min(a, b);
		}

		__host__ __device__ friend Vec3 max(const Vec3& a, const Vec3& b)
		{
			return Vec3::max(a, b);
		}
		__host__ __device__ float length() const {
			return sqrt(lengthSquared());
		}

		__host__ __device__ float lengthSquared() const {
			return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
		}

		__host__ __device__ bool nearZero() const { // Return true if the vector is close to zero in all dimensions.
			auto s = 1e-8f;
			return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
		}
	};
	class Vec2
	{
	public:
		float e[2];

		__host__ __device__ Vec2() : e{ 0, 0 } {}
		__host__ __device__ Vec2(float e0, float e1) : e{ e0, e1 } {}
		__host__ __device__ Vec2(float e) : e{ e, e } {}

		__host__ __device__ float x() const { return e[0]; }
		__host__ __device__ float y() const { return e[1]; }

		__host__ __device__ Vec2 operator-() const { return Vec2(-e[0], -e[1]); }
		__host__ __device__ float operator[](int i) const { return e[i]; }
		__host__ __device__ float& operator[](int i) { return e[i]; }

		__host__ __device__ bool operator==(const Vec2& v) const {
			return e[0] == v.e[0] && e[1] == v.e[1];
		}

		__host__ __device__ bool operator!=(const Vec2& v) const {
			return !(*this == v);
		}

		__host__ __device__ Vec2& operator+=(const Vec2& v) {
			e[0] += v.e[0];
			e[1] += v.e[1];
			return *this;
		}

		__host__ __device__ Vec2& operator*=(const Vec2& v) {
			e[0] *= v.e[0];
			e[1] *= v.e[1];
			return *this;
		}

		__host__ __device__ Vec2& operator-=(const Vec2& v) {
			e[0] -= v.e[0];
			e[1] -= v.e[1];
			return *this;
		}

		__host__ __device__ Vec2& operator*=(float t) {
			e[0] *= t;
			e[1] *= t;
			return *this;
		}

		__host__ __device__ Vec2& operator/=(float t) {
			return *this *= 1 / t;
		}

		__host__ __device__ float length() const {
			return sqrt(lengthSquared());
		}

		__host__ __device__ float lengthSquared() const {
			return e[0] * e[0] + e[1] * e[1];
		}

		__host__ __device__ bool nearZero() const {
			auto s = 1e-8f;
			return (fabs(e[0]) < s) && (fabs(e[1]) < s);
		}
	};
	using Point3 = Vec3;
	using Color = Vec3;
	// Vector Utility Functions

	__host__ __device__ inline Vec3 operator+(const Vec3& u, const Vec3& v) {
		return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
	}

	__host__ __device__ inline Vec3 operator-(const Vec3& u, const Vec3& v) {
		return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& u, const Vec3& v) {
		return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
	}

	__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
		return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
	}

	__host__ __device__ inline Vec3 operator+(const Vec3& v, float t) {
		return Vec3(v.e[0] + t, v.e[1] + t, v.e[2] + t);
	}

	__host__ __device__ inline Vec3 operator+(float t, const Vec3& v) {
		return v + t;  // Reuse the previous operator
	}

	__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) {
		return Vec3(v.e[0] * t, v.e[1] * t, v.e[2] * t);
	}

	__host__ __device__ inline Vec3 operator/(const Vec3& v, float t) {
		return (1 / t) * v;
	}

	inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
		return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
	}

	__host__ __device__ float clamp(float x, float min, float max) {
		return x < min ? min : (x > max ? max : x);
	}

	__host__ __device__ static inline Vec3 unitVector(const Vec3& v) {
		return v / v.length();
	}
	__host__ __device__ static inline float dot(const Vec3& u, const Vec3& v) {
		return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
	}

	__host__ __device__ static inline Vec3 cross(const Vec3& u, const Vec3& v) {
		return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
			u.e[2] * v.e[0] - u.e[0] * v.e[2],
			u.e[0] * v.e[1] - u.e[1] * v.e[0]);
	}
	__host__ __device__ static inline Vec3 reflect(const Vec3& v, const Vec3& n)
	{
		return v - 2 * dot(v, n) * n;
	}
	__host__ __device__ static inline Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
		float cos_theta = fminf(dot(-uv, n), 1.0f);
		Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
		Vec3 r_out_parallel = -sqrt(fabsf(1.0f - r_out_perp.lengthSquared())) * n;
		return r_out_perp + r_out_parallel;
	}
}