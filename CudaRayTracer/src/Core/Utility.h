#pragma once
#include"Core.h"
namespace BRT
{
    static class Utility
    {
    public:
        //Constans
        static const double s_Infinity;

        //Functions
        static double RandomDouble(double min, double max);

        static double RandomDouble();

        static glm::vec3 RandomVector();

        static glm::vec3 RandomVector(double min, double max);

        static glm::vec3 RandomPointInUnitSphere();

        static glm::vec3 RandomPointInUnitDisk();

        static glm::vec3 RandomPointInHemisphere(const glm::vec3& normal);

        static glm::vec3 RandomUnitVector();

        static glm::vec3 Refract(const glm::vec3& incidentRay, const glm::vec3& normal, float ior);

      
    };
}