#pragma once
#include "Core/Core.h"



namespace BRT 
{
    class Ray
    {
    public:
        Ray() = default;
        Ray(const glm::vec3& m_Origin, const glm::vec3& m_Direction)
            : m_Origin(m_Origin), m_Direction(m_Direction) {}

        glm::vec3 PointAtDistance(float distance) const;
        float  IntersectsSphere(const glm::vec3& center, float radius) const;
        

        const glm::vec3& GetOrigin() const { return m_Origin; }
        const glm::vec3& GetDirection() const { return m_Direction; }

    private:
        glm::vec3 m_Origin;
        glm::vec3 m_Direction;
    };

}