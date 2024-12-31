//#include "Hittable.cuh"
//
//namespace CRT
//{
//	class Quad : public Hittable
//	{
//		__device__ Quad(const Point3& q, const Vec3& u, const Vec3& v, int matIndex)
//			: m_Origin(q), m_U(u), m_V(v), m_MaterialIndex(matIndex)
//
//		{
//			Vec3 n = cross(u, v);
//			m_Normal = unitVector(std::move(n));
//			m_PlaneOffset = dot(m_Normal, m_Origin);
//			w = n / dot(n, n);
//		}
//
//		__device__ virtual void setBoundingBox() {
//			auto bbox_diagonal1 = AABB(m_Origin, m_Origin + m_U + m_V);
//			auto bbox_diagonal2 = AABB(m_Origin + m_U, m_Origin + m_V);
//			m_BoundingBox = AABB(bbox_diagonal1, bbox_diagonal2);
//		}
//
//		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
//		{
//			auto denom = dot(m_Normal, r.direction());
//
//			if (std::fabs(denom) < 1e-8)
//				return false;
//
//			if (std::fabs(denom) < 1e-8)
//				return false;
//
//			auto t = (m_PlaneOffset - dot(m_Normal, r.origin())) / denom;
//			if (!ray_t.contains(t))
//				return false;
//
//			auto intersection = r.pointAtDistance(t);
//
//			rec.IntersectionTime = t;
//			rec.Point = intersection;
//			rec.MaterialIndex = m_MaterialIndex;
//			rec.setFaceNormal(r, m_Normal);
//
//			return true;
//
//
//		}
//
//	private:
//		Point3 m_Origin;
//		Vec3 m_U;
//		Vec3 m_V;
//		Vec3 m_Normal;
//		int m_MaterialIndex;
//		double m_PlaneOffset;
//	};
//}