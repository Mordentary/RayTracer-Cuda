#pragma once
#include "Hittable.cuh"
#include "BVHNode.cuh"

struct Vertex
{
	CRT::Vec3 Position;
	CRT::Vec3 Normal;
	CRT::Vec2 UV;
};

namespace CRT
{
	class  Mesh : public Hittable
	{
	public:
		__device__ Mesh()
			: m_Vertices(nullptr), m_Indices(nullptr), m_VertexCount(0), m_IndexCount(0),
			m_VertexOffset(0), m_IndexOffset(0), m_MaterialIndex(0)
		{


		}

		__device__ Mesh(Vertex* globalVertices, uint32_t* globalIndices, uint32_t vertexCount, uint32_t indexCount,
			uint32_t vertexOffset, uint32_t indexOffset, uint32_t matIndx, curandState* d_rand_state)
			: m_Vertices(globalVertices + vertexOffset),
			m_Indices(globalIndices + indexOffset),
			m_VertexCount(vertexCount),
			m_IndexCount(indexCount),
			m_VertexOffset(vertexOffset),
			m_IndexOffset(indexOffset),
			m_MaterialIndex(matIndx),
			m_RandState(d_rand_state)

		{

			Vec3 min = m_Vertices[0].Position;
			Vec3 max = min;

			for (uint32_t i = 1; i < m_VertexCount; ++i)
			{
				const Vec3& pos = m_Vertices[i].Position;
				min = Vec3(fminf(min.x(), pos.x()), fminf(min.y(), pos.y()), fminf(min.z(), pos.z()));
				max = Vec3(fmaxf(max.x(), pos.x()), fmaxf(max.y(), pos.y()), fmaxf(max.z(), pos.z()));
			}

			m_BoundingBox = AABB(min, max);

		}

		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			bool hit_anything = false;
			float closest_so_far = ray_t.Max;

			for (int i = 0; i < m_IndexCount; i += 3)
			{
				const Vertex& v0 = m_Vertices[m_Indices[i] - m_VertexOffset];
				const Vertex& v1 = m_Vertices[m_Indices[i + 1] - m_VertexOffset];
				const Vertex& v2 = m_Vertices[m_Indices[i + 2] - m_VertexOffset];

				if (rayTriangleIntersect(r, v0, v1, v2, ray_t, rec))
				{
					hit_anything = true;
					closest_so_far = rec.IntersectionTime;
					ray_t.Max = closest_so_far;
				}
			}

			return hit_anything;

		}

	private:
		Vertex* m_Vertices;
		uint32_t* m_Indices;
		uint32_t m_VertexCount, m_IndexCount;
		uint32_t m_VertexOffset, m_IndexOffset;
		uint32_t m_MaterialIndex;
		curandState* m_RandState;
		BVHNode* m_MeshBVH;
	private:


		__device__ void buildBVH()
		{
			StackEntry stack[MAX_STACK_SIZE];
			int stack_top = 0;
			int next_node_index = 0;

			// Push initial work to stack
			stack[stack_top++] = { 0, num_objects, 0 };

			while (stack_top > 0)
			{
				// Pop work from stack
				StackEntry current = stack[--stack_top];
				int start = current.start;
				int end = current.end;
				int node_index = current.node_index;

				new (&nodes[node_index]) BVHNode();
				BVHNode& node = nodes[node_index];

				int object_span = end - start;

				node.m_World = objects;
				node.m_SceneNodes = nodes;

				node.m_BoundingBox = objects->m_Objects[start]->boundingBox();
				for (int i = start + 1; i < end; i++)
					node.m_BoundingBox = AABB(node.m_BoundingBox, objects->m_Objects[i]->boundingBox());

				if (object_span == 1)
				{
					// Leaf node
					node.m_ObjectIndex = start;
					node.m_ObjectCount = 1;
					node.m_IsLeaf = true;
					//node.m_BoundingBox = objects->m_Objects[start]->boundingBox();
				}
				else
				{
					//int axis = Utility::randomInt(0, 2, rand_state);

					int axis = node.m_BoundingBox.longestAxis();

					//if (object_span > 2)
					//{
					//	insertionSort(objects->m_Objects + start, object_span, axis);
					//}

					int mid = start + object_span / 2;

					// Sort objects if needed

					int left_child = ++next_node_index;
					int right_child = ++next_node_index;

					node.m_Left = left_child;
					node.m_Right = right_child;
					node.m_IsLeaf = false;

					stack[stack_top++] = { mid, end, right_child };
					stack[stack_top++] = { start, mid, left_child };
				}
			}

		}
		__device__ void calculateBoundingBox()
		{

		}

		__device__ bool rayTriangleIntersect(
			const Ray& r, const Vertex& v0, const Vertex& v1, const Vertex& v2,
			const Interval& ray_t, HitInfo& rec) const
		{
			Vec3 e1 = v1.Position - v0.Position;
			Vec3 e2 = v2.Position - v0.Position;
			Vec3 h = cross(r.direction(), e2);
			float a = dot(e1, h);

			if (fabs(a) < 1e-8f) return false;

			float f = 1.0f / a;
			Vec3 s = r.origin() - v0.Position;
			float u = f * dot(s, h);

			if (u < 0.0f || u > 1.0f) return false;

			Vec3 q = cross(s, e1);
			float v = f * dot(r.direction(), q);

			if (v < 0.0f || u + v > 1.0f) return false;

			float t = f * dot(e2, q);

			if (ray_t.surrounds(t))
			{
				rec.IntersectionTime = t;
				rec.Point = r.pointAtDistance(t);
				//rec.MaterialIndex = Utility::randomInt(3, 499, m_RandState);
				rec.MaterialIndex = m_MaterialIndex;
				// Interpolate normal
				//Vec3 normal = (1 - u - v) * v0.Normal + u * v1.Normal + v * v2.Normal;

				Vec3 normal = cross(e1, e2);
				rec.setFaceNormal(r, unitVector(normal));

				// Interpolate texture coordinates (if needed)
				 //rec.u = (1-u-v) * v0.UV.x() + u * v1.UV.x() + v * v2.UV.x();
				 //rec.v = (1-u-v) * v0.UV.y() + u * v1.UV.y() + v * v2.UV.y();

				return true;
			}

			return false;
		}
	};
}