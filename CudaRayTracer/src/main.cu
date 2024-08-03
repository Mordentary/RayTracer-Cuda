#include <SFML/Graphics.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "Core/vec3.cuh"
#include"Core/ray.cuh"
#include"Core/camera.cuh"
#include"Core/BHVNode.cuh"
#include"Core/Sphere.cuh"
#include"Core/utility.cuh"
#include"Core/Material.cuh"

namespace CRT
{
	__device__ inline Vec3 linearToGamma(const Color& linearColor)
	{
		float r = linearColor.x();
		float g = linearColor.y();
		float b = linearColor.z();

		if (r > 0)
			r = sqrt(r);
		if (g > 0)
			g = sqrt(g);
		if (b > 0)
			b = sqrt(b);

		return Vec3(r, g, b);
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
		Vec3 unit_direction = unitVector(r.getDirection());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
	}

	__device__ Color rayColor(const Ray& r, BVHNode* d_sceneRoot, HittableList* d_world, curandState* rand_state) {
		Ray current_ray = r;
		Color accumulated_color(1.0f, 1.0f, 1.0f);
		Color final_color(0.0f, 0.0f, 0.0f);
		const int MAX_BOUNCES = 50;
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
		curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);
	}

	__global__ void createRandomWorld(HittableList* world, int* objectsNum, curandState* rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			// Ground material
			int ground_material_index = world->addMaterial(new Lambertian(Color(0.5, 0.5, 0.5)));
			world->add(new Sphere(Vec3(0, -1000, 0), 1000, ground_material_index));

			for (int a = -11; a < 11; a++) {
				for (int b = -11; b < 11; b++) {
					float choose_mat = Utility::randomFloat(rand_state);
					Vec3 center(a + 0.9f * Utility::randomFloat(rand_state), 0.2f, b + 0.9f * Utility::randomFloat(rand_state));

					if ((center - Vec3(4, 0.2, 0)).length() > 0.9f) {
						int sphere_material_index;

						if (choose_mat < 0.8f) {
							// Diffuse
							Vec3 albedo = Utility::randomVector(rand_state) * Utility::randomVector(rand_state);
							sphere_material_index = world->addMaterial(new Lambertian(albedo));
						}
						else if (choose_mat < 0.95f) {
							// Metal
							Vec3 albedo = Utility::randomVector(0.5f, 1.0f, rand_state);
							float fuzz = Utility::randomFloat(0, 0.5f, rand_state);
							sphere_material_index = world->addMaterial(new Metal(albedo, fuzz));
						}
						else {
							// Glass
							sphere_material_index = world->addMaterial(new Dielectric(1.5f));
						}

						world->add(new Sphere(center, 0.2f, sphere_material_index));
					}
				}
			}

			// Add three larger spheres
			int material1 = world->addMaterial(new Dielectric(1.5f));
			world->add(new Sphere(Vec3(0, 1, 0), 1.0f, material1));

			int material2 = world->addMaterial(new Lambertian(Color(0.4, 0.2, 0.1)));
			world->add(new Sphere(Vec3(-4, 1, 0), 1.0f, material2));

			int material3 = world->addMaterial(new Metal(Color(0.7, 0.6, 0.5), 0.0f));
			world->add(new Sphere(Vec3(4, 1, 0), 1.0f, material3));
	
	
			*objectsNum = world->m_NumObjects;
		}
	}

	__global__ void createBVH(HittableList* objects, int num_objects, BVHNode* nodes, curandState* rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
			new (nodes) BVHNode(objects, nodes, rand_state);
	}

	__global__ void render(unsigned char* data, Camera* camera, BVHNode* d_sceneRoot, HittableList* d_world, curandState* rand_state, int imageWidth, int imageHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= imageWidth || y >= imageHeight) return;

		int pixel_index = y * imageWidth + x;
		curandState local_rand_state = rand_state[pixel_index];

		int data_pixel_index = (y * imageWidth + x) * 4;

		Color pixel_color(0, 0, 0);
		for (int sample = 0; sample < camera->m_SamplesPerPixel; sample++) {
			const Ray& r = camera->getRay(x, y, imageWidth, imageHeight, &local_rand_state);
			pixel_color += rayColor(r, d_sceneRoot, d_world, &local_rand_state);
		}

		rand_state[pixel_index] = local_rand_state;

		writeColor(data, data_pixel_index, camera->m_PixelSampleScale * pixel_color);
	}
}

// Global constants
constexpr float ASPECT_RATIO = 16.0f / 9.0f;
constexpr int WIDTH = 720;
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

//int main()
//{
//	const float ASPECT_RATIO = 16.0f / 9.0f;
//	int WIDTH = 720;
//	int HEIGHT = static_cast<int>(WIDTH / ASPECT_RATIO);
//	HEIGHT = (HEIGHT < 1) ? 1 : HEIGHT;
//
//	CRT::Vec3 cameraPosition(0, 3, 0);
//	CRT::Vec3 cameraTarget(0, 0, -1);
//	CRT::Vec3 worldY(0, 1, 0);
//	float aperture = 0.0001f;
//	float cameraSpeed = 0.1f;
//
//	CRT::Camera h_camera(ASPECT_RATIO, 120.0f, cameraPosition, cameraTarget, worldY, aperture, float((cameraPosition - cameraTarget).length()));
//
//	CRT::Camera* d_camera;
//	CUDA_CHECK(cudaMalloc(&d_camera, sizeof(CRT::Camera)));
//
//	CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));
//
//	const int colorChannels = 4;
//	const size_t dataImageSize = WIDTH * HEIGHT * colorChannels;
//
//	unsigned char* d_imageData;
//	CUDA_CHECK(cudaMallocManaged(&d_imageData, dataImageSize));
//
//	dim3 threads(16, 16);
//	dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);
//	int threadsPerBlock = threads.x * threads.y;
//	int totalBlocks = blocks.x * blocks.y;
//	int totalThreads = totalBlocks * threadsPerBlock;
//
//	//CRT::Hittable** d_sphere;
//	//CRT::Material** d_materials;
//	//CUDA_CHECK(cudaMalloc(&d_sphere, 4 * sizeof(CRT::Hittable*)));
//	//CUDA_CHECK(cudaMalloc(&d_materials, 3 * sizeof(CRT::Material*)));
//	//CRT::initWorld CUDA_KERNEL(1, 1)(d_world, d_sphere, d_camera, d_materials);
//	//CUDA_CHECK(cudaGetLastError());
//	//CUDA_CHECK(cudaDeviceSynchronize());
//
//// Call the global function to create the random world
//
//	// Don't forget to synchronize after the kernel launch
//	cudaDeviceSynchronize();
//
//	curandState* d_rand_state;
//	CUDA_CHECK(cudaMalloc(&d_rand_state, totalThreads * sizeof(curandState)));
//	unsigned long long seed = rand();
//	CRT::initRandState CUDA_KERNEL(blocks, threads)(d_rand_state, seed, WIDTH, HEIGHT);
//	CUDA_CHECK(cudaGetLastError());
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	CRT::HittableList* d_world;
//	CUDA_CHECK(cudaMalloc(&d_world, sizeof(CRT::HittableList)));
//
//	int* d_objectsNum;
//	cudaMallocManaged(&d_objectsNum, sizeof(int));
//
//	CRT::createRandomWorld CUDA_KERNEL(1, 1)(d_world, d_objectsNum, d_rand_state);
//	CUDA_CHECK(cudaGetLastError());
//	CUDA_CHECK(cudaDeviceSynchronize());
//	// Allocate memory for all possible nodes
//
//
//	int objectsNum = *d_objectsNum;
//	int max_nodes = 2 * objectsNum - 1;
//	CRT::BVHNode* nodes = new CRT::BVHNode[max_nodes];
//
//	// Copy to device
//	CRT::BVHNode* d_sceneNodes;
//	cudaMalloc(&d_sceneNodes, max_nodes * sizeof(CRT::BVHNode));
//	cudaMemcpy(d_sceneNodes, nodes, max_nodes * sizeof(CRT::BVHNode), cudaMemcpyHostToDevice);
//
//	CRT::createBVH CUDA_KERNEL(1, 1)(d_world, objectsNum, d_sceneNodes, d_rand_state);
//	CUDA_CHECK(cudaGetLastError());
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	delete nodes;
//	// Launch kernel to build BVH
//	//CRT::buildBVH CUDA_KERNEL(1, 1)(d_world, objectsNum, d_sceneNodes, d_rand_state);
//
//
//	//CRT::BVHNode* root;
//	//CUDA_CHECK(cudaMallocManaged(&root, sizeof(CRT::BVHNode)));
//	//CRT::BVHNode root = new CRT::BVHNode(d_WorldNode);
//
//	CRT::render CUDA_KERNEL(blocks, threads)(d_imageData, d_camera, d_sceneNodes, d_world, d_rand_state, WIDTH, HEIGHT);
//	CUDA_CHECK(cudaGetLastError());
//	CUDA_CHECK(cudaDeviceSynchronize());
//
//	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Raytracer Output");
//	sf::Texture texture;
//	texture.create(WIDTH, HEIGHT);
//	sf::Sprite sprite(texture);
//
//	sf::Image image;
//	image.create(WIDTH, HEIGHT, d_imageData);
//	image.flipVertically();
//	texture.update(image);
//
//	sf::Clock clock;
//	const float targetFPS = 60.0f;
//	const sf::Time targetFrameTime = sf::seconds(1.0f / targetFPS);
//
//	window.setMouseCursorVisible(true);
//	window.setMouseCursorGrabbed(false);
//
//	bool isRightMousePressed = false;
//
//	while (window.isOpen())
//	{
//		sf::Time elapsedTime = clock.getElapsedTime();
//		float deltaTime = elapsedTime.asSeconds();
//		clock.restart();
//
//		// Handle events
//		sf::Event event;
//		while (window.pollEvent(event))
//		{
//			if (event.type == sf::Event::Closed)
//				window.close();
//			//else if (event.type == sf::Event::Resized)
//
//			else if (event.type == sf::Event::MouseButtonPressed)
//			{
//				if (event.mouseButton.button == sf::Mouse::Right)
//				{
//					isRightMousePressed = true;
//					window.setMouseCursorVisible(false);
//					//window.setMouseCursorGrabbed(true);
//				}
//			}
//			else if (event.type == sf::Event::MouseButtonReleased)
//			{
//				if (event.mouseButton.button == sf::Mouse::Right)
//				{
//					isRightMousePressed = false;
//					window.setMouseCursorVisible(true);
//					//window.setMouseCursorGrabbed(false);
//					sf::Mouse::setPosition({ WIDTH / 2,HEIGHT / 2 }, window);
//				}
//			}
//			else if (event.type == sf::Event::KeyPressed)
//			{
//				if (event.key.code == sf::Keyboard::PageUp)
//				{
//					h_camera.adjustFocusDistance(0.1f);
//					std::cout << "Focus Distance: " << h_camera.getFocusDistance() << std::endl;
//				}
//				else if (event.key.code == sf::Keyboard::PageDown)
//				{
//					h_camera.adjustFocusDistance(-0.1f);
//					std::cout << "Focus Distance: " << h_camera.getFocusDistance() << std::endl;
//				}
//			}
//		}
//
//		// Update camera
//		sf::Vector2i mousePos = sf::Mouse::getPosition(window);
//		h_camera.updateCamera(deltaTime, WIDTH, HEIGHT,
//			static_cast<float>(mousePos.x), static_cast<float>(mousePos.y),
//			isRightMousePressed);
//
//		// Update camera on GPU
//		CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));
//
//		// Re-render the scene
//		CRT::render CUDA_KERNEL(blocks, threads)(d_imageData, d_camera, d_sceneNodes, d_world, d_rand_state, WIDTH, HEIGHT);
//		CUDA_CHECK(cudaGetLastError());
//		CUDA_CHECK(cudaDeviceSynchronize());
//
//		// Update the texture and draw
//		sf::Image image;
//		image.create(WIDTH, HEIGHT, d_imageData);
//		image.flipVertically();
//		texture.update(image);
//
//		window.clear();
//		window.draw(sprite);
//		window.display();
//	}
//
//	// Clean up (after the loop ends)
//	CUDA_CHECK(cudaFree(d_camera));
//	CUDA_CHECK(cudaFree(d_world));
//	CUDA_CHECK(cudaFree(d_rand_state));
//	CUDA_CHECK(cudaFree(d_imageData));
//
//	return 0;
//}

void initializeScene(CRT::Camera* d_camera, curandState* d_rand_state, CRT::HittableList* d_world,
	int* d_objectsNum, const RenderConfig& config) {
	unsigned long long seed = rand();
	CRT::initRandState CUDA_KERNEL(config.blocks, config.threads)(d_rand_state, seed, WIDTH, HEIGHT);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CRT::createRandomWorld CUDA_KERNEL(1, 1)(d_world, d_objectsNum, d_rand_state);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

CRT::BVHNode* createBVH(CRT::HittableList* d_world, int objectsNum, curandState* d_rand_state) {
	int max_nodes = 2 * objectsNum - 1;
	CRT::BVHNode* d_sceneNodes;
	CUDA_CHECK(cudaMalloc(&d_sceneNodes, max_nodes * sizeof(CRT::BVHNode)));

	CRT::createBVH CUDA_KERNEL(1, 1)(d_world, objectsNum, d_sceneNodes, d_rand_state);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	return d_sceneNodes;
}

void renderScene(unsigned char* d_imageData, CRT::Camera* d_camera, CRT::BVHNode* d_sceneNodes,
	CRT::HittableList* d_world, curandState* d_rand_state, const RenderConfig& config) {
	CRT::render CUDA_KERNEL(config.blocks, config.threads)(d_imageData, d_camera, d_sceneNodes, d_world, d_rand_state, WIDTH, HEIGHT);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void handleEvents(WindowConfig& windowConfig, bool& isRightMousePressed, CRT::Camera& h_camera) {
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

void updateAndRender(WindowConfig& windowConfig, CRT::Camera& h_camera, CRT::Camera* d_camera,
	unsigned char* d_imageData, CRT::BVHNode* d_sceneNodes, CRT::HittableList* d_world,
	curandState* d_rand_state, const RenderConfig& config, float deltaTime) {
	sf::Vector2i mousePos = sf::Mouse::getPosition(windowConfig.window);
	
	h_camera.updateCamera(deltaTime, WIDTH, HEIGHT, static_cast<float>(mousePos.x), static_cast<float>(mousePos.y), sf::Mouse::isButtonPressed(sf::Mouse::Right));

	CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));

	renderScene(d_imageData, d_camera, d_sceneNodes, d_world, d_rand_state, config);
}

void drawFrame(WindowConfig& windowConfig, unsigned char* d_imageData) {
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
		CRT::Vec3 cameraPosition(0, 3, 0);
		CRT::Vec3 cameraTarget(0, 0, -1);
		CRT::Vec3 worldY(0, 1, 0);

		CRT::Camera h_camera(ASPECT_RATIO, FOV, cameraPosition, cameraTarget, worldY, APERTURE, (cameraPosition - cameraTarget).length());

		RenderConfig renderConfig{
			dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
			dim3((WIDTH + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, (HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
			0
		};
		renderConfig.totalThreads = renderConfig.blocks.x * renderConfig.blocks.y * renderConfig.threads.x * renderConfig.threads.y;

		WindowConfig windowConfig;

		// Use RAII for CUDA memory management
		CudaMemory<CRT::Camera> d_camera(sizeof(CRT::Camera));
		CudaMemory<unsigned char> d_imageData(WIDTH * HEIGHT * 4, true);
		CudaMemory<curandState> d_rand_state(renderConfig.totalThreads * sizeof(curandState));
		CudaMemory<CRT::HittableList> d_world(sizeof(CRT::HittableList));
		CudaMemory<int> d_objectsNum(sizeof(int), true);

		CUDA_CHECK(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));

		initializeScene(d_camera, d_rand_state, d_world, d_objectsNum, renderConfig);

		int objectsNum = *d_objectsNum;
		CudaMemory<CRT::BVHNode> d_sceneNodes((2 * objectsNum - 1) * sizeof(CRT::BVHNode));
		CRT::createBVH CUDA_KERNEL(1, 1)(d_world, objectsNum, d_sceneNodes, d_rand_state);
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
			updateAndRender(windowConfig, h_camera, d_camera, d_imageData, d_sceneNodes, d_world, d_rand_state, renderConfig, deltaTime);
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