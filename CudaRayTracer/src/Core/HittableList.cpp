#include "HittableList.h"


namespace BRT
{
    bool HittableList::Hit(const Ray& ray, double tMin, double tMax, HitInfo& rec) const {
        HitInfo temp_rec;
        bool hit_anything = false;
        double closest_so_far = tMax;

        for (const auto& object : m_HittableObjects) {
            if (object->Hit(ray, tMin, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.IntersectionTime;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
}