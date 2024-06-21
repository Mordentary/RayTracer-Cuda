#pragma once
#include"Core.h"
#include"Hittable.h"
#include <vector>

namespace BRT 
{
	class HittableList : Hittable
	{

        public:
            HittableList() = default;
            HittableList(Shared<Hittable> object) { AddObject(object); }

            void ClearList() { m_HittableObjects.clear(); }
            void AddObject(Shared<Hittable> object) { m_HittableObjects.push_back(object); }

            virtual bool Hit(const Ray& ray, double tMin, double tMax, HitInfo& info) const override;


        private:
            std::vector<Shared<Hittable>> m_HittableObjects;
    };

}

