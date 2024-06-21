#include "Camera.h"

#include"Utility.h"
namespace BRT
{
    //DefaultCamera 
    Camera::Camera(double aspectRatio, double fov, float viewportHeight, glm::vec3 position, glm::vec3 target, glm::vec3 up, double aperture,
        float focusDist) :
        m_Origin(position)
    {
        auto theta = glm::radians(fov);
        auto h = std::tan(theta / 2);

        float vpHeight =  h * 2;
        float vpWidth = vpHeight * aspectRatio;

         m_CameraZ = glm::normalize(position - target);
         m_CameraX = glm::normalize(glm::cross(m_CameraZ, up));
         m_CameraY = glm::cross(m_CameraX, m_CameraZ);

        m_vpHorizontalOffset = m_CameraX * vpWidth * focusDist;
        m_vpVerticalOffset  = m_CameraY * vpHeight * focusDist;

        m_LensRadius = aperture / 2;
            
        m_vpLowerLeftCorner = m_Origin - m_vpHorizontalOffset / 2.f - m_vpVerticalOffset / 2.f - focusDist * m_CameraZ;
       

    }



    Ray Camera::GetRay(float x, float y) const
    {

        glm::vec3 rd = m_LensRadius * Utility::RandomPointInUnitDisk();
        glm::vec3 offset = m_CameraX * rd.x + m_CameraY * rd.y;

        return { m_Origin + offset, m_vpLowerLeftCorner + x * m_vpHorizontalOffset + y * m_vpVerticalOffset - m_Origin - offset};
    }



   /* Camera::Camera(double aspectRatio, const glm::vec2& viewportSize, const glm::vec3& origin, double focalLenght, double fov)
        : 
        m_Origin(origin), 
        m_vpHorizontalOffset(glm::vec3(viewportSize.x, 0.0, 0.0)), 
        m_vpVerticalOffset(glm::vec3(0.0, viewportSize.y, 0.0))
    {

        auto theta = glm::radians(fov);
        auto h = std::tan(theta / 2);
        m_vpVerticalOffset.y *= h;
        m_vpHorizontalOffset.x = aspectRatio * m_vpVerticalOffset.y;
        m_vpLowerLeftCorner = origin - m_vpHorizontalOffset.x / 2 - m_vpVerticalOffset.y / 2 - glm::vec3(0, 0, focalLenght);
    }*/
}