#include"Utility.h"
#include<random>
#include <glm/gtx/norm.hpp>
namespace BRT 
{
        const double Utility::s_Infinity = std::numeric_limits<double>::infinity();

        double Utility::RandomDouble(double min, double max)
        {
            std::uniform_real_distribution<double> distribution(min, max);
            static std::mt19937 generator;
            return distribution(generator);
        }

        double Utility::RandomDouble()
        {
            static std::uniform_real_distribution<double> distribution(0.0, 1.0);
            static std::mt19937 generator;
            return distribution(generator);
        }


        inline  glm::vec3 Utility::RandomVector() {
            return { RandomDouble(), RandomDouble(), RandomDouble() };
        }

        inline  glm::vec3 Utility::RandomVector(double min, double max) {
            return { RandomDouble(min, max), RandomDouble(min, max), RandomDouble(min, max) };
        }


        glm::vec3 Utility::RandomPointInUnitSphere()
        {
            while (true)
            {
                glm::vec3 point = RandomVector(-1, 1);
                if (glm::length2(point) >= 1) continue;
                return point;
            }
        }


        glm::vec3 Utility::RandomPointInUnitDisk() {
            while (true) {
                glm::vec3 point = glm::vec3(RandomDouble(-1, 1), RandomDouble(-1, 1), 0);
                if (glm::length2(point) >= 1) continue;
                return point;
            }
        }

        glm::vec3 Utility::RandomPointInHemisphere(const glm::vec3& normal)
        {
            glm::vec3 point = RandomPointInUnitSphere();
            if (dot(point, normal) > 0.0f) // In the same hemisphere as the normal
                return point;
            else
                return -point;
        }



        glm::vec3 Utility::RandomUnitVector()
        {
            return glm::normalize(RandomPointInUnitSphere());
        }
    
        glm::vec3 Utility::Refract(const glm::vec3& incidentRay, const glm::vec3& normal, float eta)
        {
            float cos_theta = std::fmin(glm::dot(-incidentRay, normal), 1.0);
            glm::vec3 perp = eta * (incidentRay + cos_theta * normal);
            glm::vec3 parallel =  (float)- std::sqrt(std::fabs(1.0 - glm::length2(perp))) * normal;
            return perp + parallel;
        }


}