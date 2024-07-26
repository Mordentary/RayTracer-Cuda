#pragma once
#include "Hittable.cuh"

namespace CRT
{
	class HittableList : public Hittable
	{
	public:
		static const int s_MAX_OBJECTS = 500;
		Hittable* m_Objects[s_MAX_OBJECTS];
		int m_NumObjects;

		__device__ HittableList() : m_NumObjects(0)
		{
			for (int i = 0; i < s_MAX_OBJECTS; i++) {
				m_Objects[i] = nullptr;
			}
		}

		__device__ void add(Hittable* object) {
			if (m_NumObjects < s_MAX_OBJECTS) {
				m_Objects[m_NumObjects++] = object;
			}
		}

		__device__ bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override {
			HitInfo temp_rec;
			bool hit_anything = false;
			float closest_so_far = ray_t.max;


			for (int i = 0; i < m_NumObjects; i++) {
				if (m_Objects[i] && m_Objects[i]->hit(r, Interval(ray_t.min, closest_so_far), temp_rec))
				{
					hit_anything = true;
					closest_so_far = temp_rec.IntersectionTime;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}
	};
}