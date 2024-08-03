#include"HittableList.cuh"

namespace CRT
{
	class BVHNode : public Hittable
	{
	public:
		__host__ __device__ BVHNode() = default;

		__device__ BVHNode(HittableList* list, BVHNode* preallocatedNodes, curandState* rand_state) : m_SceneNodes(preallocatedNodes), m_World(list)
		{
			buildBVH(list, list->m_NumObjects, preallocatedNodes, rand_state);
		}

#define MAX_STACK_SIZE 64  // Adjust based on your maximum tree depth

		struct StackEntry {
			int start;
			int end;
			int node_index;
		};

		__device__ void buildBVH(HittableList* objects, int num_objects, BVHNode* nodes, curandState* rand_state)
		{
			StackEntry stack[MAX_STACK_SIZE];
			int stack_top = 0;
			int next_node_index = 1;  // 0 is root

			// Push initial work to stack
			stack[stack_top++] = { 0, num_objects, 0 };



			while (stack_top > 0)
			{
				// Pop work from stack
				StackEntry current = stack[--stack_top];
				int start = current.start;
				int end = current.end;
				int node_index = current.node_index;

				BVHNode& node = nodes[node_index];
				int object_span = end - start;

				 node.m_BoundingBox = objects->m_Objects[start]->boundingBox();
				for (int i = start + 1; i < end; i++)
					node.m_BoundingBox = AABB(node.m_BoundingBox, objects->m_Objects[i]->boundingBox());

				if (object_span == 1)
				{
					// Leaf node
					node.m_ObjectIndex = start;
					node.m_ObjectCount = 1;
					node.m_IsLeaf = true;
					node.m_BoundingBox = objects->m_Objects[start]->boundingBox();
				}
				else
				{
					//int axis = Utility::randomInt(0, 2, rand_state);

				 int axis = node.m_BoundingBox.longestAxis();


					int mid = start + object_span / 2;

					// Sort objects if needed
					if (object_span > 2)
					{
						insertionSort(objects->m_Objects + start, object_span, axis);
					}

					// Set up this node
					int left_child = next_node_index++;
					int right_child = next_node_index++;

					node.m_Left = left_child;
					node.m_Right = right_child;
					node.m_IsLeaf = false;

					// Push right child work to stack
					stack[stack_top++] = { mid, end, right_child };
					// Push left child work to stack
					stack[stack_top++] = { start, mid, left_child };
				}
			}

		}

		__device__ void insertionSort(Hittable** arr, int n, int axis) {
			for (int i = 1; i < n; i++) {
				Hittable* key = arr[i];
				int j = i - 1;

				while (j >= 0 && boxCompare(key, arr[j], axis)) {
					arr[j + 1] = arr[j];
					j = j - 1;
				}
				arr[j + 1] = key;
			}
		}

		struct StackItem
		{
			int nodeIndex;
			Interval t_interval;
		};
		__device__ virtual bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			// Stack for traversing the BVH tree

			StackItem stack[16];
			int stackSize = 1;

			// Start with the root node (index 0	
			stack[0] = { 0, ray_t };

			bool hitAnything = false;
			HitInfo tempRec;

			while (stackSize > 0)
			{
				StackItem current = stack[--stackSize];
				int currentIndex = current.nodeIndex;
				Interval currentInterval = current.t_interval;

				if (currentIndex < 0 || currentIndex >= m_World->m_NumObjects * 2 - 1) continue;
				// Check if the ray intersects with the current node's bounding box
				if (!m_SceneNodes[currentIndex].m_BoundingBox.hit(r, currentInterval))
					continue;

				if (m_SceneNodes[currentIndex].m_IsLeaf)
				{
					// Leaf node: test intersection with the object
					for (int i = 0; i < m_SceneNodes[currentIndex].m_ObjectCount; ++i)
					{
						int objectIndex = m_SceneNodes[currentIndex].m_ObjectIndex + i;
						if (objectIndex < 0 || objectIndex >= m_World->m_NumObjects) continue;

						if (m_World->m_Objects[objectIndex]->hit(r, currentInterval, tempRec))
						{
							hitAnything = true;
							currentInterval.Max = tempRec.IntersectionTime;
							rec = tempRec;
						}
					}
				}
				else
				{
					// Internal node: add child nodes to the stack
					if (stackSize < (16) - 2)
					{
						// Right child
						stack[stackSize++] = { m_SceneNodes[currentIndex].m_Right, currentInterval };
						// Left child (always the next node in the array)
						stack[stackSize++] = { m_SceneNodes[currentIndex].m_Left, currentInterval };
					}
				}
			}

			return hitAnything;
		}
	private:
		int m_Left;
		int m_Right;
		int m_ObjectIndex;  // For leaf nodes
		int m_ObjectCount;  // For leaf nodes
		bool m_IsLeaf;
		AABB m_BoundingBox;
		BVHNode* m_SceneNodes;
		HittableList* m_World;  // Pointer to the array of objects

		__device__ static bool boxCompare(const Hittable* a, const Hittable* b, int axis_index) {
			auto a_axis_interval = a->boundingBox().axisInterval(axis_index);
			auto b_axis_interval = b->boundingBox().axisInterval(axis_index);
			return a_axis_interval.Min < b_axis_interval.Min;
		}

		__device__ static bool boxCompareX(const Hittable* a, const Hittable* b) {
			return boxCompare(a, b, 0);
		}

		__device__ static bool boxCompareY(const Hittable* a, const Hittable* b) {
			return boxCompare(a, b, 1);
		}

		__device__ static bool boxCompareZ(const Hittable* a, const Hittable* b) {
			return boxCompare(a, b, 2);
		}
	};
}