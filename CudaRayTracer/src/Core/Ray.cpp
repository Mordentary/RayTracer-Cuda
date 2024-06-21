#include "Ray.h"

namespace BRT
{
    glm::vec3 Ray::PointAtDistance(float distance) const
    {
        return m_Origin + distance * m_Direction;
    }

    float Ray::IntersectsSphere(const glm::vec3& center, float radius) const
    {
        glm::vec3 to_center = center - m_Origin;
        float distance_to_center = glm::length(to_center);
        float projection = glm::dot(to_center, m_Direction);
        float sphere_intersection = radius * radius - (distance_to_center * distance_to_center - projection * projection);
        return sphere_intersection;
    }
}