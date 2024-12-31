#pragma once
#include "Hittable.cuh"
#include "BVHNode.cuh"
#include"AABB.cuh"
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

		__device__ Mesh(Vertex* globalVertices,
			uint32_t* globalIndices,
			int* globalFaceMaterialIds,
			uint32_t  vertexCount,
			uint32_t  indexCount,
			uint32_t  vertexOffset,
			uint32_t  indexOffset,
			uint32_t  faceMatOffset,
			curandState* d_rand_state)
			: m_Vertices(globalVertices + vertexOffset),
			m_Indices(globalIndices + indexOffset),
			m_FaceMaterialIds(globalFaceMaterialIds + faceMatOffset),
			m_VertexCount(vertexCount),
			m_IndexCount(indexCount),
			m_VertexOffset(vertexOffset),
			m_IndexOffset(indexOffset),
			m_FaceMatOffset(faceMatOffset),
			m_RandState(d_rand_state)
		{
			if (m_VertexCount > 0)
			{
				m_BoundingBox = AABB_EMPTY;
				for (uint32_t i = 0; i < m_VertexCount; ++i)
				{
					const Vec3& pos = m_Vertices[i].Position;
					m_BoundingBox.expand(pos);
				}
			}
			else
			{
				m_BoundingBox = AABB();
			}
			buildBVHMesh();
		}

		__device__ virtual inline bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			bool hitAnything = false;
			float closestSoFar = ray_t.max;
			// Traversal stack
			uint32_t stack[MAX_STACK_SIZE]{};
			int stackPtr = 0;
			int nodeIdx = 0;  // Start with the root

			while (true)
			{
				const BVHNode& node = m_MeshBVH[nodeIdx];
				HitInfo tempRec;

				if (node.m_BoundingBox.hit(r, Interval(ray_t.min, closestSoFar), tempRec))
				{
					if (!node.m_IsLeaf)
					{
						stack[stackPtr++] = node.m_Right;
						nodeIdx = node.m_Left;
						continue;
					}

					//// Check if this AABB hit is closer than any previous hit
					//if (tempRec.IntersectionTime < closestSoFar)
					//{
					//	// Mark this as an AABB hit
					//	tempRec.MaterialIndex = 1;
					//	rec = tempRec;
					//	hitAnything = true;
					//	closestSoFar = tempRec.IntersectionTime;
					//}

					// Test intersection with triangles in this leaf node

					for (int i = node.m_ObjectIndex; i < node.m_ObjectIndex + node.m_ObjectCount; i += 3)
					{
						const Vertex& v0 = m_Vertices[m_Indices[i]];
						const Vertex& v1 = m_Vertices[m_Indices[i + 1]];
						const Vertex& v2 = m_Vertices[m_Indices[i + 2]];

						if (rayTriangleIntersect(r, v0, v1, v2, Interval(ray_t.min, closestSoFar), tempRec, i))
						{
							hitAnything = true;
							closestSoFar = tempRec.IntersectionTime;
							ray_t.max = closestSoFar;
							rec = tempRec;
						}
					}
				}

				if (stackPtr == 0) break;
				nodeIdx = stack[--stackPtr];
			}
			return hitAnything;
		}

	private:
		Vertex* m_Vertices;
		uint32_t* m_Indices;
		int* m_FaceMaterialIds;
		uint32_t m_VertexCount, m_IndexCount;
		uint32_t m_VertexOffset, m_IndexOffset, m_FaceMatOffset;
		curandState* m_RandState;
		BVHNode* m_MeshBVH;
	private:
		__device__ void buildBVHMesh() {
			int numTriangles = m_IndexCount / 3;
			int maxNodes = 2 * numTriangles - 1;
			m_MeshBVH = new BVHNode[maxNodes]; // Freed when ~HittableList or ~Mesh is called

			// Basic stack-based approach
			struct StackEntry { int start, end, nodeIndex; };
			StackEntry stack[MAX_STACK_SIZE];
			int stackTop = 0, nextNodeIndex = 0;

			// Root node
			BVHNode& root = m_MeshBVH[nextNodeIndex++];
			root.m_BoundingBox = m_BoundingBox; // full box
			root.m_ObjectIndex = 0;
			root.m_ObjectCount = m_IndexCount; // all indices
			root.m_IsLeaf = false;

			stack[stackTop++] = { 0, (int)m_IndexCount, 0 };

			while (stackTop > 0) {
				StackEntry cur = stack[--stackTop];
				int start = cur.start;
				int end = cur.end;
				int nodeIdx = cur.nodeIndex;

				BVHNode& node = m_MeshBVH[nodeIdx];
				int span = end - start;
				if (span <= 24 || (nextNodeIndex + 1) >= maxNodes) {
					// Leaf (3 triangles or fewer)
					node.m_IsLeaf = true;
					node.m_ObjectIndex = start;
					node.m_ObjectCount = span;
					// set bounding box precisely
					node.m_BoundingBox = computeTrianglesAABB(start, span);
					continue;
				}

				// find best axis & split
				int bestAxis = 0;
				float bestPos = 0.f;
				float bestCost = 1e30f;

				// Compute axis bounding
				for (int axis = 0; axis < 3; axis++) {
					float minPos = 1e30f;
					float maxPos = -1e30f;
					for (int i = start; i < end; i += 3) {
						CRT::Vec3 c = triangleCentroid(i);
						if (c[axis] < minPos) minPos = c[axis];
						if (c[axis] > maxPos) maxPos = c[axis];
					}
					float midPos = 0.5f * (minPos + maxPos);
					float cost = evaluateSAH(node, axis, midPos, start, end);
					if (cost < bestCost) {
						bestCost = cost;
						bestAxis = axis;
						bestPos = midPos;
					}
				}

				// partition
				int mid = start;
				for (int i = start; i < end; i += 3) {
					CRT::Vec3 c = triangleCentroid(i);
					if (c[bestAxis] < bestPos) {
						// swap triangles in m_Indices
						swap_triplet(m_Indices[mid], m_Indices[mid + 1], m_Indices[mid + 2],
							m_Indices[i], m_Indices[i + 1], m_Indices[i + 2]);
						// also swap faceMaterialIds
						int faceLeft = mid / 3;
						int faceRight = i / 3;
						int tmp = m_FaceMaterialIds[faceLeft];
						m_FaceMaterialIds[faceLeft] = m_FaceMaterialIds[faceRight];
						m_FaceMaterialIds[faceRight] = tmp;

						mid += 3;
					}
				}

				// create child nodes
				node.m_Left = nextNodeIndex++;
				node.m_Right = nextNodeIndex++;
				node.m_IsLeaf = false;

				// push children
				stack[stackTop++] = { mid, end,   node.m_Right };
				stack[stackTop++] = { start, mid, node.m_Left };
			}

			// finalize bounding boxes bottom-up
			for (int i = nextNodeIndex - 1; i >= 0; i--) {
				BVHNode& nd = m_MeshBVH[i];
				if (!nd.m_IsLeaf) {
					AABB boxL = m_MeshBVH[nd.m_Left].m_BoundingBox;
					AABB boxR = m_MeshBVH[nd.m_Right].m_BoundingBox;
					nd.m_BoundingBox = AABB::combine(boxL, boxR);
				}
			}
		}

		// Evaluate a “surface area heuristic”
		__device__ float evaluateSAH(const BVHNode& node, int axis, float pos, int start, int end) {
			AABB leftBox = AABB_EMPTY;
			AABB rightBox = AABB_EMPTY;
			int leftCount = 0, rightCount = 0;

			for (int i = start; i < end; i += 3) {
				CRT::Vec3 c = triangleCentroid(i);
				if (c[axis] < pos) {
					leftCount++;
					expandTriangleAABB(leftBox, i);
				}
				else {
					rightCount++;
					expandTriangleAABB(rightBox, i);
				}
			}
			float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
			return cost < 1e-8f ? 1e-8f : cost;
		}

		__device__ void expandTriangleAABB(AABB& box, int i) {
			CRT::Vec3 p0 = m_Vertices[m_Indices[i]].Position;
			CRT::Vec3 p1 = m_Vertices[m_Indices[i + 1]].Position;
			CRT::Vec3 p2 = m_Vertices[m_Indices[i + 2]].Position;
			box.expand(p0);
			box.expand(p1);
			box.expand(p2);
		}

		__device__ CRT::Vec3 triangleCentroid(int i) {
			CRT::Vec3 p0 = m_Vertices[m_Indices[i]].Position;
			CRT::Vec3 p1 = m_Vertices[m_Indices[i + 1]].Position;
			CRT::Vec3 p2 = m_Vertices[m_Indices[i + 2]].Position;
			return (p0 + p1 + p2) * (1.f / 3.f);
		}

		__device__ AABB computeTrianglesAABB(int start, int count) {
			AABB box = AABB_EMPTY;
			for (int i = start; i < start + count; i += 3) {
				expandTriangleAABB(box, i);
			}
			return box;
		}

		__device__ bool rayTriangleIntersect(
			const Ray& r,
			const Vertex& v0,
			const Vertex& v1,
			const Vertex& v2,
			const Interval& ray_t,
			HitInfo& rec,
			int triBaseIndex) const
		{
			// Möller–Trumbore or similar
			const float EPSILON = 1e-8f;
			CRT::Vec3 edge1 = v1.Position - v0.Position;
			CRT::Vec3 edge2 = v2.Position - v0.Position;
			CRT::Vec3 h = cross(r.direction(), edge2);
			float a = dot(edge1, h);
			if (fabs(a) < EPSILON) return false;

			float f = 1.f / a;
			CRT::Vec3 s = r.origin() - v0.Position;
			float u = f * dot(s, h);
			if (u < 0.f || u > 1.f) return false;

			CRT::Vec3 q = cross(s, edge1);
			float v = f * dot(r.direction(), q);
			if (v < 0.f || (u + v) > 1.f) return false;

			float t = f * dot(edge2, q);
			if (t < ray_t.min || t > ray_t.max) return false;

			// We have an intersection
			rec.IntersectionTime = t;
			rec.Point = r.pointAtDistance(t);
			// get face index
			int faceIdx = triBaseIndex / 3;
			rec.MaterialIndex = m_FaceMaterialIds[faceIdx];

			// normal
			CRT::Vec3 normal = cross(edge1, edge2);
			normal = unitVector(normal);
			rec.setFaceNormal(r, normal);

			return true;
		}
	};
}