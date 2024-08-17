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
					//	tempRec.MaterialIndex = nodeIdx % 50; // Special index for AABBs
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

						if (rayTriangleIntersect(r, v0, v1, v2, Interval(ray_t.min, closestSoFar), tempRec, nodeIdx))
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
		//__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		//{
		//	bool hit_anything = false;
		//	float closest_so_far = ray_t.max;

		//	for (int i = 0; i < m_IndexCount; i += 3)
		//	{
		//		const Vertex& v0 = m_Vertices[m_Indices[i] - m_VertexOffset];
		//		const Vertex& v1 = m_Vertices[m_Indices[i + 1] - m_VertexOffset];
		//		const Vertex& v2 = m_Vertices[m_Indices[i + 2] - m_VertexOffset];

		//		if (rayTriangleIntersect(r, v0, v1, v2, ray_t, rec))
		//		{
		//			hit_anything = true;
		//			closest_so_far = rec.IntersectionTime;
		//			ray_t.max = closest_so_far;
		//		}
		//	}

		//	return hit_anything;
		//}

	private:
		Vertex* m_Vertices;
		uint32_t* m_Indices;
		uint32_t m_VertexCount, m_IndexCount;
		uint32_t m_VertexOffset, m_IndexOffset;
		uint32_t m_MaterialIndex;
		curandState* m_RandState;
		BVHNode* m_MeshBVH;
	private:
		__device__ float evaluateSAH(const BVHNode& node, int axis, float pos, int start, int end)
		{
			AABB leftBox = AABB_EMPTY;
			AABB rightBox = AABB_EMPTY;
			int leftCount = 0, rightCount = 0;

			for (uint32_t i = start; i < end; i += 3)  // Assuming triangles
			{
				int idx = node.m_ObjectIndex + i;
				const Vertex& v0 = m_Vertices[m_Indices[idx]];
				const Vertex& v1 = m_Vertices[m_Indices[idx + 1]];
				const Vertex& v2 = m_Vertices[m_Indices[idx + 2]];

				// Calculate triangle centroid
				Vec3 centroid = (v0.Position + v1.Position + v2.Position) * (1.0f / 3.0f);

				if (centroid[axis] < pos)
				{
					leftCount++;
					leftBox.expand(AABB(v0.Position, v1.Position));
					leftBox.expand(v2.Position);
				}
				else
				{
					rightCount++;
					rightBox.expand(AABB(v0.Position, v1.Position));
					rightBox.expand(v2.Position);
				}
			}

			float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
			return cost > 0 ? cost : FLT_MAX;
		}

		struct StackEntryMesh {
			int start;
			int end;
			int node_index;
			int parent_indices_span;
		};
		__device__ void buildBVHMesh() {
			const int numTriangles = m_IndexCount / 3;
			const int maxNodes = 2 * numTriangles - 1;
			BVHNode* nodes = new BVHNode[maxNodes];
			m_MeshBVH = nodes;

			struct StackEntry { int start, end, nodeIndex; };
			StackEntry stack[MAX_STACK_SIZE];
			int stackTop = 0;
			int nextNodeIndex = 0;

			// Initialize root node
			BVHNode& root = nodes[nextNodeIndex++];
			root.m_BoundingBox = m_BoundingBox;
			stack[stackTop++] = { 0, (int)m_IndexCount, 0 };

			while (stackTop > 0) {
				StackEntry current = stack[--stackTop];
				BVHNode& node = nodes[current.nodeIndex];
				int indicesSpan = current.end - current.start;

				if (indicesSpan <= 12 || nextNodeIndex + 1 >= maxNodes) {  // 4 triangles or less
					// Create leaf node
					node.m_IsLeaf = true;
					node.m_ObjectIndex = current.start;
					node.m_ObjectCount = indicesSpan;
					continue;
				}

				// Find best split
				int bestAxis = -1;
				float bestPos = 0, bestCost = FLT_MAX;
				for (int axis = 0; axis < 3; ++axis) {
					float minPos = FLT_MAX, maxPos = -FLT_MAX;
					for (int i = current.start; i < current.end; i += 3) {
						Vec3 centroid = (m_Vertices[m_Indices[i]].Position +
							m_Vertices[m_Indices[i + 1]].Position +
							m_Vertices[m_Indices[i + 2]].Position) * (1.0f / 3.0f);
						minPos = minPos < centroid[axis] ? minPos : centroid[axis];
						maxPos = maxPos > centroid[axis] ? maxPos : centroid[axis];
					}
					float splitPos = (minPos + maxPos) * 0.5f;
					float cost = evaluateSAH(node, axis, splitPos, current.start, current.end);
					if (cost < bestCost) {
						bestAxis = axis;
						bestPos = splitPos;
						bestCost = cost;
					}
				}

				int mid = current.start;
				for (int i = current.start; i < current.end; i += 3) {
					Vec3 centroid = (m_Vertices[m_Indices[i]].Position +
						m_Vertices[m_Indices[i + 1]].Position +
						m_Vertices[m_Indices[i + 2]].Position) * (1.0f / 3.0f);
					if (centroid[bestAxis] < bestPos) {
						swap_triplet(m_Indices[mid], m_Indices[mid + 1], m_Indices[mid + 2],
							m_Indices[i], m_Indices[i + 1], m_Indices[i + 2]);
						mid += 3;
					}
				}

				// Create child nodes
				node.m_Left = nextNodeIndex++;
				node.m_Right = nextNodeIndex++;
				node.m_IsLeaf = false;

				// Push right child, then left child (to process left child first)
				stack[stackTop++] = { mid, current.end, node.m_Right };
				stack[stackTop++] = { current.start, mid, node.m_Left };
			}

			// Compute AABBs for all nodes bottom-up
			for (int i = maxNodes - 1; i >= 0; --i) {
				BVHNode& node = nodes[i];
				if (node.m_IsLeaf) {
					node.m_BoundingBox = computeTrianglesAABB(node.m_ObjectIndex, node.m_ObjectCount);
				}
				else {
					node.m_BoundingBox = nodes[node.m_Left].m_BoundingBox;
					node.m_BoundingBox.expand(nodes[node.m_Right].m_BoundingBox);
				}
			}
		}

		__device__ AABB computeTrianglesAABB(int start, int count) {
			AABB box = AABB_EMPTY;
			for (int i = start; i < start + count; i += 3) {
				box.expand(m_Vertices[m_Indices[i]].Position);
				box.expand(m_Vertices[m_Indices[i + 1]].Position);
				box.expand(m_Vertices[m_Indices[i + 2]].Position);
			}
			return box;
		}
		__device__ int getApproximateLevel(int nodeIndex) const {
			nodeIndex++; // Adjust for 0-based index
			int level = 0;
			while (nodeIndex > 1) {
				nodeIndex >>= 1; // Equivalent to nodeIndex /= 2
				level++;
			}
			return level;
		}

		__device__ bool rayTriangleIntersect(
			const Ray& ray, const Vertex& v0, const Vertex& v1, const Vertex& v2,
			const Interval& ray_t, HitInfo& rec, int nodeIndex) const
		{
			const float EPSILON = 1e-7f;
			Vec3 edge1 = v1.Position - v0.Position;
			Vec3 edge2 = v2.Position - v0.Position;
			Vec3 h = cross(ray.direction(), edge2);
			float a = dot(edge1, h);

			// Check if ray is parallel to the triangle
			if (fabs(a) < EPSILON)
				return false;

			float f = 1.0f / a;
			Vec3 s = ray.origin() - v0.Position;
			float u = f * dot(s, h);

			if (u < -EPSILON || u > 1.0f + EPSILON)
				return false;

			Vec3 q = cross(s, edge1);
			float v = f * dot(ray.direction(), q);

			if (v < -EPSILON || u + v > 1.0f + EPSILON)
				return false;

			float t = f * dot(edge2, q);

			if (t >= ray_t.min && t <= ray_t.max) {
				rec.IntersectionTime = t;
				rec.Point = ray.pointAtDistance(t);
				rec.MaterialIndex = m_MaterialIndex;

				 //Interpolate normal
					//Vec3 normal = (1 - u - v) * v0.Normal + u * v1.Normal + v * v2.Normal;
					//rec.setFaceNormal(ray, (normal));
					
					rec.setFaceNormal(ray, unitVector(cross(edge1, edge2)));

				return true;
			}

			return false;
		}
	};
}