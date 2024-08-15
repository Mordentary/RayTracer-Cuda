#include <SFML/Graphics.hpp>
#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include"tiny_obj_loader.h"

#include "Core/Vec3.cuh"
#include"Core/Ray.cuh"
#include"Core/Camera.cuh"
#include"Core/BVHNode.cuh"
#include"Core/Sphere.cuh"
#include"Core/Utility.cuh"
#include"Core/Material.cuh"
#include"Core/Mesh.cuh"

namespace CRT
{
	__constant__ CRT::Camera d_camera;

	__device__ inline Vec3 linearToGamma(const Color& linearColor)
	{
		return Vec3(sqrt(linearColor.x()), sqrt(linearColor.y()), sqrt(linearColor.z()));
	}

	__device__ inline double linearToGamma(double component)
	{
		if (component > 0)
			return sqrt(component);
		return 0;
	}

	__device__ inline void writeColor(unsigned char* data, int index, const Vec3& color)
	{
		//Vec3 colorInGamma = linearToGamma(color);
		float r = linearToGamma(color.x());
		float g = linearToGamma(color.y());
		float b = linearToGamma(color.z());

		Interval defaultIntensity(0.000f, 0.999f);
		data[index] = static_cast<unsigned char>(256 * defaultIntensity.clamp(r));
		data[index + 1] = static_cast<unsigned char>(256 * defaultIntensity.clamp(g));
		data[index + 2] = static_cast<unsigned char>(256 * defaultIntensity.clamp(b));
		data[index + 3] = 255;
	}

	__device__ inline Color getBackgroundColor(const Ray& r) {
		Vec3 unit_direction = unitVector(r.direction());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
	}

	__device__ Color rayColor(const Ray& r, BVHNode* d_sceneRoot, HittableList* d_world, curandState* rand_state) {
		Ray current_ray = r;
		Color accumulated_color(1.0f, 1.0f, 1.0f);
		Color final_color(0.0f, 0.0f, 0.0f);
		const int MAX_BOUNCES = 10;
		const int MIN_BOUNCES = 3;
		const float MAX_PROB = 0.95f;

		for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
			HitInfo rec;

			if (bounce >= MIN_BOUNCES) {
				float survival_prob = fmaxf(accumulated_color.x(), fmaxf(accumulated_color.y(), accumulated_color.z()));
				survival_prob = fminf(survival_prob, MAX_PROB);

				if (curand_uniform(rand_state) > survival_prob) {
					break;  // Terminate the path
				}
				accumulated_color /= survival_prob;
			}

			if (d_sceneRoot->hit(current_ray, Interval(0.001f, INFINITY), rec)) {
				Ray scattered{};
				Color attenuation{};

				if (rec.MaterialIndex >= 0 && rec.MaterialIndex < d_world->m_NumMaterials &&
					d_world->m_Materials[rec.MaterialIndex]->Scatter(current_ray, rec, attenuation, scattered, rand_state)) {
					accumulated_color *= attenuation;
					current_ray = scattered;
				}
				else {
					return Color(0, 0, 0);
				}
			}
			else {
				// Ray didn't hit anything, add background color
				Color background_color = getBackgroundColor(current_ray);
				final_color = accumulated_color * background_color;
				break;
			}
		}
		return final_color;
	}

	__global__ void initRandState(curandState* rand_state, unsigned long long seed, int width, int height) {
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x >= width || y >= height) return;
		int pixel_index = y * width + x;

		// Use a unique seed for each pixel
		unsigned long long pixel_seed = seed + pixel_index;
		curand_init(pixel_seed, pixel_index, 0, &rand_state[pixel_index]);
	}
	__device__ Color getLevelColor(int nodeLevel) {
		const int maxLevels = 50; // Adjust based on your maximum expected tree depth
		nodeLevel = nodeLevel % maxLevels; // Ensure we don't exceed our color array

		// Define a set of distinct colors
		const Color colors[] = {
			Color(1.0f, 0.0f, 0.0f),   // Red
			Color(0.0f, 1.0f, 0.0f),   // Green
			Color(0.0f, 0.0f, 1.0f),   // Blue
			Color(1.0f, 1.0f, 0.0f),   // Yellow
			Color(1.0f, 0.5f, 0.0f),   // Orange
			Color(0.5f, 0.0f, 1.0f),   // Purple
			Color(0.0f, 0.5f, 0.0f),   // Dark Green
		};

		const int numColors = sizeof(colors) / sizeof(colors[0]);

		// For deeper levels, alternate between full intensity and half intensity
		Color baseColor = colors[nodeLevel % numColors];
		if ((nodeLevel / numColors) % 2 == 1) {
			baseColor = baseColor * 0.5f;
		}

		return baseColor;
	}
	__global__ void createRandomWorld(HittableList* world, Mesh* meshes, int* objectsNum, curandState* rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			new (world) CRT::HittableList();
			//Ground material
			//const int MAX_DEBUG_LEVELS = 50;
			//// Create debug materials
			//for (int i = 0; i < MAX_DEBUG_LEVELS; i++) {
			//	Color levelColor = getLevelColor(i);
			//	world->addMaterial(new Lambertian(levelColor));
			//}

			world->add(meshes);

			int ground_material_index = world->addMaterial(new Lambertian(Color(0.5, 0.5, 0.5)));
			world->add(new Sphere(Vec3(0, -1000, 0), 999, ground_material_index));

			//for (int a = -11; a < 11; a++) {
			//	for (int b = -11; b < 11; b++) {
			//		float choose_mat = Utility::randomFloat(rand_state);
			//		Vec3 center(a + 0.9f * Utility::randomFloat(rand_state), 0.2f, b + 0.9f * Utility::randomFloat(rand_state));

			//		if ((center - Vec3(4, 0.2, 0)).length() > 0.9f) {
			//			int sphere_material_index;

			//			if (choose_mat < 0.8f) {
			//				// Diffuse
			//				Vec3 albedo = Utility::randomVector(rand_state) * Utility::randomVector(rand_state);
			//				sphere_material_index = world->addMaterial(new Lambertian(albedo));
			//			}
			//			else if (choose_mat < 0.95f) {
			//				// Metal
			//				Vec3 albedo = Utility::randomVector(0.5f, 1.0f, rand_state);
			//				float fuzz = Utility::randomFloat(0, 0.5f, rand_state);
			//				sphere_material_index = world->addMaterial(new Metal(albedo, fuzz));
			//			}
			//			else {
			//				// Glass
			//				sphere_material_index = world->addMaterial(new Dielectric(1.5f));
			//			}

			//			world->add(new Sphere(center, 0.2f, sphere_material_index));
			//		}
			//	}
			//}

			//// Add three larger spheres

			int material2 = world->addMaterial(new Lambertian(Color(0.4, 0.2, 0.1)));
			world->add(new Sphere(Vec3(-4, 3, 0), 0.5f, material2));

			int material3 = world->addMaterial(new Metal(Color(0.7, 0.6, 0.5), 0.0f));
			world->add(new Sphere(Vec3(4, 1, 0), 1.0f, material3));

			for (int i = 0; i < world->s_MAX_MATERIALS; i++) {
				// Generate a random color (albedo)
				Color albedo = Utility::randomVector(0.2f, 1.0f, rand_state);

				// Generate a random roughness between 0 (perfectly smooth) and 1 (very rough)
				float roughness = Utility::randomFloat(0.0f, 0.1f, rand_state);

				// Create a new Metal material with the random albedo and roughness
				int materialIndex = world->addMaterial(new Lambertian(albedo));
			}

			//world->add(new Sphere(Vec3(0, 1, 0), 1.0f, material1));

			*objectsNum = world->m_NumObjects;
		}
	}

	__global__ void createBVH(HittableList* objects, BVHNode* nodes, curandState* rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			//new (nodes) BVHNode(objects, nodes, rand_state);

			BVHNode::buildBVHScene(objects, objects->m_NumObjects, nodes, rand_state);
		}
	}
	__global__ void initMesh(Mesh* mesh, Vertex* globalVertices, uint32_t* globalIndices,
		uint32_t vertexCount, uint32_t indexCount,
		uint32_t vertexOffset, uint32_t indexOffset, curandState* d_rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			new (mesh) Mesh(globalVertices, globalIndices, vertexCount, indexCount, vertexOffset, indexOffset, 0, d_rand_state);
		}
	}
	__device__ void debugRandomDistribution(curandState* rand_state, unsigned char* image, int width, int height) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if ((x >= width) || (y >= height)) return;
		int pixel_index = y * width + x;

		curandState local_rand_state = rand_state[pixel_index];
		float r = curand_uniform(&local_rand_state);

		float color = static_cast<float>(r * 256);
		image[pixel_index * 4 + 0] = color;  // R
		image[pixel_index * 4 + 1] = color;  // G
		image[pixel_index * 4 + 2] = color;  // B
		image[pixel_index * 4 + 3] = 255;    // A


		rand_state[pixel_index] = local_rand_state;
	}
	__global__ void render(unsigned char* data, BVHNode* d_sceneRoot, HittableList* d_world, curandState* rand_state, int imageWidth, int imageHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= imageWidth || y >= imageHeight) return;

		//debugRandomDistribution(rand_state, data, imageWidth, imageHeight);
		int pixel_index = y * imageWidth + x;
		curandState local_rand_state = rand_state[pixel_index];

		int data_pixel_index = pixel_index * 4;

		Color pixel_color(0, 0, 0);
		for (int sample = 0; sample < d_camera.m_SamplesPerPixel; sample++) 
		{
			const Ray& r = d_camera.getRay(x, y, imageWidth, imageHeight, &local_rand_state);
			pixel_color += rayColor(r, d_sceneRoot, d_world, &local_rand_state);
		}


		writeColor(data, data_pixel_index, d_camera.m_PixelSampleScale * pixel_color);
		rand_state[pixel_index] = local_rand_state;
	}
}

// Global constants
constexpr float ASPECT_RATIO = 16.0f / 9.0f;
constexpr int WIDTH = 1440;
constexpr int HEIGHT = static_cast<int>(WIDTH / ASPECT_RATIO) < 1 ? 1 : static_cast<int>(WIDTH / ASPECT_RATIO);
constexpr float APERTURE = 0.0001f;
constexpr float FOV = 90.0f;
constexpr int THREADS_PER_BLOCK = 16;
// Configuration structs
struct RenderConfig {
	dim3 threads;
	dim3 blocks;
	int totalThreads;
};

struct WindowConfig {
	sf::RenderWindow window;
	sf::Texture texture;
	sf::Sprite sprite;

	WindowConfig() : window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Raytracer Output") {
		if (!texture.create(WIDTH, HEIGHT)) {
			throw std::runtime_error("Failed to create SFML texture");
		}
		sprite.setTexture(texture);
	}
};

void loadObject(const std::string& filename,
	std::vector<Vertex>& vertices,
	std::vector<uint32_t>& indices)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str());
	if (!err.empty()) {
		std::cerr << "Error: " << err << std::endl;
	}
	if (!ret) {
		throw std::runtime_error("Failed to load object file");
	}

	vertices.clear();
	indices.clear();
	std::unordered_map<std::string, uint32_t> uniqueVertices;

	// Variables for normalization
	float minX = std::numeric_limits<float>::max(), minY = minX, minZ = minX;
	float maxX = std::numeric_limits<float>::lowest(), maxY = maxX, maxZ = maxX;

	// Process all shapes
	for (const auto& shape : shapes) {
		const auto& mesh = shape.mesh;

		for (const auto& index : mesh.indices) {
			Vertex vertex;

			// Position
			vertex.Position[0] = attrib.vertices[3 * index.vertex_index + 0];
			vertex.Position[1] = attrib.vertices[3 * index.vertex_index + 1];
			vertex.Position[2] = attrib.vertices[3 * index.vertex_index + 2];

			// Update min and max for normalization
			minX = std::min(minX, vertex.Position[0]);
			minY = std::min(minY, vertex.Position[1]);
			minZ = std::min(minZ, vertex.Position[2]);
			maxX = std::max(maxX, vertex.Position[0]);
			maxY = std::max(maxY, vertex.Position[1]);
			maxZ = std::max(maxZ, vertex.Position[2]);

			vertex.Position *= 3;
			// Normal
			if (index.normal_index >= 0) {
				vertex.Normal[0] = attrib.normals[3 * index.normal_index + 0];
				vertex.Normal[1] = attrib.normals[3 * index.normal_index + 1];
				vertex.Normal[2] = attrib.normals[3 * index.normal_index + 2];
			}
			else {
				vertex.Normal[0] = vertex.Normal[1] = vertex.Normal[2] = 0.0f;
			}

			// Texture coordinates
			if (index.texcoord_index >= 0) {
				vertex.UV[0] = attrib.texcoords[2 * index.texcoord_index + 0];
				vertex.UV[1] = attrib.texcoords[2 * index.texcoord_index + 1];
			}
			else {
				vertex.UV[0] = vertex.UV[1] = 0.0f;
			}

			// Create a string key for the vertex
			std::string key = std::to_string(vertex.Position[0]) + "," +
				std::to_string(vertex.Position[1]) + "," +
				std::to_string(vertex.Position[2]) + "," +
				std::to_string(vertex.Normal[0]) + "," +
				std::to_string(vertex.Normal[1]) + "," +
				std::to_string(vertex.Normal[2]) + "," +
				std::to_string(vertex.UV[0]) + "," +
				std::to_string(vertex.UV[1]);

			if (uniqueVertices.count(key) == 0) {
				uniqueVertices[key] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vertex);
			}
			indices.push_back(uniqueVertices[key]);
		}
	}

	// Normalize vertices
	float centerX = (minX + maxX) / 2.0f;
	float centerY = (minY + maxY) / 2.0f;
	float centerZ = (minZ + maxZ) / 2.0f;
	float maxDim = std::max({ maxX - minX, maxY - minY, maxZ - minZ });
	float scale = 2.0f / maxDim;

	for (auto& vertex : vertices) {
		vertex.Position[0] = (vertex.Position[0] - centerX) * scale;
		vertex.Position[1] = (vertex.Position[1] - centerY) * scale;
		vertex.Position[2] = (vertex.Position[2] - centerZ) * scale;
	}

	std::cout << "Loaded and normalized " << vertices.size() << " vertices and "
		<< indices.size() << " indices from " << shapes.size() << " shapes." << std::endl;
}

//struct MeshData {
//	CudaMem<CRT::Mesh> d_meshes;
//	CudaMem<Vertex> d_vertices;
//	CudaMem<uint32_t> d_indices;
//	MeshData(CudaMem<CRT::Mesh>&& meshes, CudaMem<Vertex>&& vertices, CudaMem<uint32_t>&& indices)
//		: d_meshes(std::move(meshes)), d_vertices(std::move(vertices)), d_indices(std::move(indices)) {}
//};
//
//static std::unique_ptr<MeshData> d_meshData;
void initMeshes(CudaMem<CRT::Mesh>& d_meshes, CudaMem<Vertex>& d_vertices, CudaMem<uint32_t>& d_indices, curandState* d_rand_state)
{
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	try {
		loadObject("assets/models/cornell-box.obj", vertices, indices);
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading object: " << e.what() << std::endl;
	}

	d_vertices.reallocate(vertices.size());
	d_indices.reallocate(indices.size());

	// Copy vertex and index data to GPU
	d_vertices.copyFromHost(vertices.data(), vertices.size());
	d_indices.copyFromHost(indices.data(), indices.size());

	CRT::initMesh CUDA_KERNEL(1, 1)(d_meshes, d_vertices, d_indices, vertices.size(), indices.size(), 0, 0, d_rand_state);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}
void initializeScene(curandState* d_rand_state, CRT::HittableList* d_world, CudaMem<CRT::Mesh>& d_meshes, CudaMem<Vertex>& vertices, CudaMem<uint32_t>& indices,
	int* d_objectsNum, const RenderConfig& config) {
	unsigned long long seed = rand();
	CRT::initRandState CUDA_KERNEL(config.blocks, config.threads)(d_rand_state, seed, WIDTH, HEIGHT);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	initMeshes(d_meshes, vertices, indices, d_rand_state);

	CRT::createRandomWorld CUDA_KERNEL(1, 1)(d_world, d_meshes, d_objectsNum, d_rand_state);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

//CRT::BVHNode* createBVH(CRT::HittableList* d_world, int objectsNum, curandState* d_rand_state) {
//	int max_nodes = 2 * objectsNum - 1;
//	CudaMem<CRT::BVHNode> d_sceneNodes(max_nodes);
//
//	CRT::createBVH CUDA_KERNEL(1, 1)(d_world, d_sceneNodes, d_rand_state);
//	CUDA_CHECK(cudaGetLastError());
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	return d_sceneNodes;
//}

void renderScene(unsigned char* d_imageData, CRT::BVHNode* d_sceneNodes,
	CRT::HittableList* d_world, curandState* d_rand_state, const RenderConfig& config)

{
	CRT::render CUDA_KERNEL(config.blocks, config.threads)(d_imageData, d_sceneNodes, d_world, d_rand_state, WIDTH, HEIGHT);
	//CRT::render CUDA_KERNEL(config.blocks, config.threads)(d_imageData, d_sceneNodes, d_world, d_rand_state, WIDTH, HEIGHT);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void handleEvents(WindowConfig& windowConfig, bool& isRightMousePressed, CRT::Camera& h_camera)
{
	sf::Event event;
	while (windowConfig.window.pollEvent(event)) {
		if (event.type == sf::Event::Closed)
			windowConfig.window.close();
		else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Right) {
			isRightMousePressed = true;
			windowConfig.window.setMouseCursorVisible(false);
		}
		else if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Right) {
			isRightMousePressed = false;
			windowConfig.window.setMouseCursorVisible(true);
			sf::Mouse::setPosition({ WIDTH / 2, HEIGHT / 2 }, windowConfig.window);
		}
		else if (event.type == sf::Event::KeyPressed) {
			if (event.key.code == sf::Keyboard::PageUp)
				h_camera.adjustFocusDistance(0.1f);
			else if (event.key.code == sf::Keyboard::PageDown)
				h_camera.adjustFocusDistance(-0.1f);
		}
	}
}

void updateAndRender(WindowConfig& windowConfig, CRT::Camera& h_camera,
	unsigned char* d_imageData, CRT::BVHNode* d_sceneNodes, CRT::HittableList* d_world,
	curandState* d_rand_state, const RenderConfig& config, float deltaTime)
{
	sf::Vector2i mousePos = sf::Mouse::getPosition(windowConfig.window);

	h_camera.updateCamera(deltaTime, WIDTH, HEIGHT, static_cast<float>(mousePos.x), static_cast<float>(mousePos.y), sf::Mouse::isButtonPressed(sf::Mouse::Right));

	COPY_TO_SYMBOL(CRT::d_camera, &h_camera, sizeof(CRT::Camera));
	//CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));

	renderScene(d_imageData, d_sceneNodes, d_world, d_rand_state, config);
}

void drawFrame(WindowConfig& windowConfig, unsigned char* d_imageData)
{
	sf::Image image;
	image.create(WIDTH, HEIGHT, d_imageData);
	image.flipVertically();
	windowConfig.texture.update(image);

	windowConfig.window.clear();
	windowConfig.window.draw(windowConfig.sprite);
	windowConfig.window.display();
}

int main() {
	try {
		CRT::Vec3 cameraPosition(0, 4, 4);
		CRT::Vec3 cameraTarget(0, 0, 0);
		CRT::Vec3 worldY(0, 1, 0);

		CRT::Camera h_camera(ASPECT_RATIO, FOV, cameraPosition, cameraTarget, worldY, APERTURE, (cameraPosition - cameraTarget).length());

		RenderConfig renderConfig{
			dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
			dim3((WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
			0
		};
		renderConfig.totalThreads = renderConfig.blocks.x * renderConfig.blocks.y * renderConfig.threads.x * renderConfig.threads.y;

		WindowConfig windowConfig;

		//Increase memory limits
		size_t size_heap, size_stack;
		//cudaError_t error = cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
		//if (error != cudaSuccess) {
		//	printf("cudaDeviceSetLimit failed with %d, line(%d)\n", error, __LINE__);
		//	exit(EXIT_FAILURE);
		//}
		//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 20000000 * sizeof(double));
		cudaDeviceSetLimit(cudaLimitStackSize, 12928);
		cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
		cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
		printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);

		CudaMem<CRT::Camera> d_camera{ 1 };
		CudaMem<unsigned char> d_imageData(WIDTH * HEIGHT * 4, true, CudaAlloc::Type::Bytes);
		CudaMem<curandState> d_rand_state(renderConfig.totalThreads);
		CudaMem<CRT::HittableList> d_world{ 1 };
		CudaMem<int> d_objectsNum(1, true);
		CudaMem<CRT::Mesh> d_meshes{ 1 };
		CudaMem<Vertex> d_vertices{ 1 };
		CudaMem<uint32_t> d_indices{ 1 };

		COPY_TO_SYMBOL(CRT::d_camera, &h_camera, sizeof(CRT::Camera));

		initializeScene(d_rand_state, d_world, d_meshes, d_vertices, d_indices, d_objectsNum, renderConfig);

		int objectsNum = *d_objectsNum;

		int max_nodes = 2 * objectsNum - 1;
		CudaMem<CRT::BVHNode> d_sceneNodes(max_nodes);

		CRT::createBVH CUDA_KERNEL(1, 1)(d_world, d_sceneNodes, d_rand_state);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());

		sf::Clock clock;
		const float targetFPS = 60.0f;
		const sf::Time targetFrameTime = sf::seconds(1.0f / targetFPS);

		bool isRightMousePressed = false;

		while (windowConfig.window.isOpen()) {
			sf::Time elapsedTime = clock.getElapsedTime();
			float deltaTime = elapsedTime.asSeconds();
			clock.restart();

			handleEvents(windowConfig, isRightMousePressed, h_camera);
			updateAndRender(windowConfig, h_camera, d_imageData, d_sceneNodes, d_world, d_rand_state, renderConfig, deltaTime);
			drawFrame(windowConfig, d_imageData);

			sf::sleep(targetFrameTime - clock.getElapsedTime());
		}

		return EXIT_SUCCESS;
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (...) {
		std::cerr << "Unknown error occurred" << std::endl;
		return EXIT_FAILURE;
	}
}