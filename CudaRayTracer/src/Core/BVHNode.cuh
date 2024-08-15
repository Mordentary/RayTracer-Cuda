#pragma once
#include"HittableList.cuh"
#include "AABB.cuh"

namespace CRT
{
#define MAX_STACK_SIZE 64

	class BVHNode : public Hittable
	{
		friend class Mesh;
	public:
		__host__ __device__ BVHNode() = default;

		struct StackEntry {
			int start;
			int end;
			int node_index;
		};

		__device__ static void buildBVHScene(HittableList* objects, int num_objects, BVHNode* nodes, curandState* rand_state)
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
				node.m_Nodes = nodes;

				node.m_BoundingBox = AABB_EMPTY;
				for (int i = start; i < end; i++)
					node.m_BoundingBox.expand(objects->m_Objects[i]->boundingBox());

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

		__device__ static void insertionSort(Hittable** arr, int n, int axis) {
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
		__device__ __forceinline__ static void optimizedSelectionSort(Hittable** arr, int n, int axis)
		{
			for (int i = 0; i < n - 1; i++) {
				int minIdx = i;
				for (int j = i + 1; j < n; j++) {
					if (boxCompare(arr[j], arr[minIdx], axis)) {
						minIdx = j;
					}
				}
				if (minIdx != i) {
					Hittable* temp = arr[i];
					arr[i] = arr[minIdx];
					arr[minIdx] = temp;
				}
			}
		}

		__device__ virtual inline bool hit(const Ray& r, Interval ray_t, HitInfo& rec) const override
		{
			bool hitAnything = false;
			float closestSoFar = ray_t.max;

			// Traversal stack
			uint32_t stack[MAX_STACK_SIZE / 2]{};
			int stackPtr = 0;
			int nodeIdx = 0;  // Start with the root

			while (true)
			{
				const BVHNode& node = m_Nodes[nodeIdx];
				HitInfo tempRec;
				if (node.m_BoundingBox.hit(r, ray_t, tempRec)) {
					if (node.m_IsLeaf) {
						//for (int i = 0; i < node.m_ObjectCount; ++i) {
						int objectIndex = node.m_ObjectIndex;
						//+ i;
						HitInfo tempRec;
						if (m_World->m_Objects[objectIndex]->hit(r, Interval(ray_t.min, closestSoFar), tempRec)) {
							hitAnything = true;
							closestSoFar = tempRec.IntersectionTime;
							rec = tempRec;
						}
						//}
						if (stackPtr == 0) break;
						nodeIdx = stack[--stackPtr];
					}
					else {
						stack[stackPtr++] = node.m_Right;
						nodeIdx = node.m_Left;
					}
				}
				else {
					if (stackPtr == 0) break;
					nodeIdx = stack[--stackPtr];
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
		BVHNode* m_Nodes;
		HittableList* m_World;

	private:
		__device__ static bool boxCompare(const Hittable* a, const Hittable* b, int axis_index) {
			auto a_axis_interval = a->boundingBox().axisInterval(axis_index);
			auto b_axis_interval = b->boundingBox().axisInterval(axis_index);
			return a_axis_interval.min < b_axis_interval.min;
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