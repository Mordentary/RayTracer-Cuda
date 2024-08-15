#pragma once
#include "Hittable.cuh"
namespace CRT
{
	class HittableList : public Hittable
	{
	public:
		__device__ HittableList() : m_NumObjects(0), m_NumMaterials(0)
		{
			for (int i = 0; i < s_MAX_OBJECTS; i++) {
				m_Objects[i] = nullptr;
			}
			for (int i = 0; i < s_MAX_MATERIALS; i++) {
				m_Materials[i] = nullptr;
			}
			m_BoundingBox = AABB();

		}

			__device__ ~HittableList() {
			for (int i = 0; i < m_NumObjects; i++) {
				if (m_Objects[i]) {
					delete m_Objects[i];
				}
			}
			for (int i = 0; i < m_NumMaterials; i++) {
				if (m_Materials[i]) {
					delete m_Materials[i];
				}
			}
		}
		__device__ HittableList(const HittableList&) = delete;
		__device__ HittableList& operator=(const HittableList&) = delete;



		__device__ void add(Hittable* object) {
			if (m_NumObjects < s_MAX_OBJECTS) {
				m_Objects[m_NumObjects++] = object;
				m_BoundingBox = AABB(m_BoundingBox, object->boundingBox());
			}
		}

		__device__ int addMaterial(Material* material) {
			if (m_NumMaterials < s_MAX_MATERIALS) {
				m_Materials[m_NumMaterials] = material;
				return m_NumMaterials++;
			}
			return -1;  // Error: too many materials
		}

		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override {
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

	public:
		static const int s_MAX_OBJECTS = 500;
		static const int s_MAX_MATERIALS = 500;

		Hittable* m_Objects[s_MAX_OBJECTS];
		int m_NumObjects;
		Material* m_Materials[s_MAX_MATERIALS];
		int m_NumMaterials;

	};
}