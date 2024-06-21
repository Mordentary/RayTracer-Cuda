#include "Material.h"
#include"Utility.h"
#include"Hittable.h"
#include <glm/gtc/epsilon.hpp>


namespace BRT 
{
    bool Matte::Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const
    {

        glm::vec3 scatterDirection = rec.Normal + Utility::RandomUnitVector();
        float tolerance = 0.001f;
        if (glm::all(glm::epsilonEqual(scatterDirection, glm::vec3(0), tolerance)))
            scatterDirection = rec.Normal;
        scattered = Ray(rec.Point, scatterDirection);
        attenuation = m_Albedo;


        return true;
    }
    bool Metal::Scatter(const Ray& rayIn, const HitInfo& rec, glm::vec3& attenuation, Ray& scattered) const
    {
        glm::vec3 reflected = glm::reflect(glm::normalize(rayIn.GetDirection()), rec.Normal);

        scattered = Ray(rec.Point, reflected + m_Roughness * Utility::RandomPointInUnitSphere());
        attenuation = m_Albedo;
        return (glm::dot(scattered.GetDirection(), rec.Normal) > 0.f);

    }

    bool Dialectric::Scatter(const Ray& rayIn, const HitInfo& info, glm::vec3& attenuation, Ray& scattered) const
    {
        attenuation = glm::vec3(1.0, 1.0, 1.0);
        float refraction_ratio = info.IsNormalOutward ? (1.0 / m_IndexOfRefraction) : m_IndexOfRefraction;

        glm::vec3 unitDirection = glm::normalize(rayIn.GetDirection());
        double cos_theta = fmin(dot(-unitDirection, info.Normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        glm::vec3 direction;

        if (cannot_refract || SchlicksReflectance(cos_theta, refraction_ratio) > Utility::RandomDouble())
            direction = glm::reflect(unitDirection, info.Normal);
        else
            direction = Utility::Refract(unitDirection, info.Normal, refraction_ratio);

     

        scattered = Ray(info.Point, direction);

        return true;

    }

    double Dialectric::SchlicksReflectance(double cosine, double ref_idx) const
    {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
}
