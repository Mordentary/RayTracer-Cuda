#pragma once
#include"Ray.h"

namespace BRT
{
	class Camera
	{
    public:

        Camera(double aspectRatio, double fov, float viewportHeight, glm::vec3 position, glm::vec3 target, glm::vec3 up, double aperture,
            float focusDist);
      //  Camera(double aspectRatio, const glm::vec2& viewportSize, const glm::vec3& origin, double focalLenght = 1.0, double fov = 90.0);
       

        Ray GetRay(float x, float y) const;
    private:
        glm::vec3 m_Origin;
        glm::vec3 m_vpLowerLeftCorner;
        glm::vec3 m_vpHorizontalOffset, m_vpVerticalOffset;

        glm::vec3 m_CameraX, m_CameraY, m_CameraZ;
        float m_LensRadius;

	};
}

