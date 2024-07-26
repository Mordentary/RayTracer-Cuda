#include <SFML/Graphics.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include "Core/vec3.cuh"
#include"Core/ray.cuh"
#include"Core/camera.cuh"
#include"Core/HittableList.cuh"
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

	//__device__ color rayColor(const Ray& r, HittableList* d_world, curandState* rand_state)
	//{
	//	Ray current_ray = r;
	//	color accumulated_color(1.0, 1.0, 1.0);
	//	color final_color(0.0, 0.0, 0.0);

	//	const int MAX_BOUNCES = 50;  // Prevent infinite loops

	//	for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
	//		HitInfo rec;

	//		if ((*d_world).hit(current_ray, Interval(0.001f, INFINITY), rec)) {
	//			// The ray hit something
	//			Vec3 direction = Utility::randomOnHemisphere(rec.Normal, rand_state);

	//			// Update the ray for the next iteration
	//			current_ray = Ray(rec.Point, direction);

	//			// Accumulate color (0.5 for each bounce)
	//			accumulated_color *= 0.5f;
	//		}
	//		else {
	//			// The ray didn't hit anything, calculate sky color
	//			Vec3 unit_direction = unitVector(current_ray.getDirection());
	//			float a = 0.5f * (unit_direction.y() + 1.0f);
	//			color sky_color = (1.0f - a) * color(1.0f, 1.0f, 1.0f) + a * color(0.5f, 0.7f, 1.0f);

	//			// Add the sky color to our final color and exit the loop
	//			final_color += accumulated_color * sky_color;
	//			break;
	//		}
	//	}

	//	return final_color;
	//}
	__device__ inline Color getBackgroundColor(const Ray& r) {
		Vec3 unit_direction = unitVector(r.getDirection());
		float t = 0.5f * (unit_direction.y() + 1.0f);
		return (1.0f - t) * Color(1.0f, 1.0f, 1.0f) + t * Color(0.5f, 0.7f, 1.0f);
	}
	__device__ Color rayColor(const Ray& r, HittableList* d_world, curandState* rand_state) {
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

			if (d_world->hit(current_ray, Interval(0.001f, INFINITY), rec)) {
				Ray scattered{};
				Color attenuation{};

				if (rec.MaterialPtr && rec.MaterialPtr->Scatter(current_ray, rec, attenuation, scattered, rand_state)) {
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

	__global__ void createRandomWorld(HittableList* world, curandState* rand_state)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			new (world) HittableList();
			// Reset the world
			world->m_NumObjects = 0;

			// Create ground
			Material* ground_material = new Lambertian(Color(0.5, 0.5, 0.5));
			world->add(new Sphere(Vec3(0, -1000, 0), 1000, ground_material));

			// Create random small spheres
			for (int a = -11; a < 11 && world->m_NumObjects < HittableList::s_MAX_OBJECTS - 3; a++) {
				for (int b = -11; b < 11 && world->m_NumObjects < HittableList::s_MAX_OBJECTS - 3; b++) {
					float choose_mat = Utility::randomFloat(rand_state);
					Vec3 center(a + 0.9f * Utility::randomFloat(rand_state), 0.2f, b + 0.9f * Utility::randomFloat(rand_state));

					if ((center - Vec3(4, 0.2, 0)).length() > 0.9f) {
						Material* sphere_material;

						if (choose_mat < 0.8f) {
							// Diffuse
							Vec3 albedo = Utility::randomVector(rand_state) * Utility::randomVector(rand_state);
							sphere_material = new Lambertian(albedo);
						}
						else if (choose_mat < 0.95f) {
							// Metal
							Vec3 albedo = Utility::randomVector(0.5f, 1.0f, rand_state);
							float fuzz = Utility::randomFloat(0, 0.5f, rand_state);
							sphere_material = new Metal(albedo, fuzz);
						}
						else {
							// Glass
							sphere_material = new Dialectric(1.5f);
						}

						world->add(new Sphere(center, 0.2f, sphere_material));

						if (world->m_NumObjects >= HittableList::s_MAX_OBJECTS - 3) {
							break;
						}
					}
				}
			}

			// Add three larger spheres if there's room
			if (world->m_NumObjects < HittableList::s_MAX_OBJECTS) {
				Material* material1 = new Dialectric(1.5f);
				world->add(new Sphere(Vec3(0, 1, 0), 1.0f, material1));
			}

			if (world->m_NumObjects < HittableList::s_MAX_OBJECTS) {
				Material* material2 = new Lambertian(Color(0.4, 0.2, 0.1));
				world->add(new Sphere(Vec3(-4, 1, 0), 1.0f, material2));
			}

			if (world->m_NumObjects < HittableList::s_MAX_OBJECTS) {
				Material* material3 = new Metal(Color(0.7, 0.6, 0.5), 0.0f);
				world->add(new Sphere(Vec3(4, 1, 0), 1.0f, material3));
			}
		}
	}
	__global__ void initWorld(HittableList* d_world, Hittable** d_hittables, Camera* camera, Material** d_materials)
	{
		if (threadIdx.x == 0 && blockIdx.x == 0)
		{
			d_materials[0] = new Lambertian(Vec3(0.5f));
			d_materials[1] = new Metal(Vec3(0.5f, 0.4f, 0.1f), 0.01f);
			d_materials[2] = new Dialectric(1.33f);
			//new (&d_sphere[0]) Sphere(Point3(0, 0, -2), 0.5, &d_materials[0]);
			//new (&d_sphere[1]) Sphere(Point3(0, -100.5, -1), 100, &d_materials[1]);
			d_hittables[0] = new Sphere(Point3(0, 0, -2), 0.5, d_materials[2]);
			d_hittables[1] = new Sphere(Point3(0, -100.5, -1), 100, d_materials[0]);
			d_hittables[2] = new Sphere(Point3(1, 0, -2), 0.4, d_materials[0]);
			d_hittables[3] = new Sphere(Point3(5, 3, 2), 0.9, d_materials[0]);
			// Add the Sphere to the list
			d_world->add(d_hittables[0]);
			d_world->add(d_hittables[1]);
			d_world->add(d_hittables[2]);
			d_world->add(d_hittables[3]);
		}
	}

	//__global__ void render(unsigned char* data, Camera* camera, HittableList* d_world, curandState* rand_state, int imageWidth, int imageHeight)
	//{
	//	int x = blockIdx.x * blockDim.x + threadIdx.x;
	//	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//	if (x >= imageWidth && y >= imageHeight) return;

	//	int idx = (blockIdx.y * gridDim.x + blockIdx.x);
	//	curandState* local_rand_state = &rand_state[idx];
	//	__shared__ curandState sharedRandState[16][16]{};
	//	sharedRandState[threadIdx.y][threadIdx.x] = rand_state[idx];

	//	int index = (y * imageWidth + x) * 4;
	//	Color pixel_color(0, 0, 0);
	//	for (int sample = 0; sample < camera->m_SamplesPerPixel; sample++) {
	//
	//		const Ray& r = camera->getRay(x, y, imageWidth, imageHeight, &sharedRandState[threadIdx.y][threadIdx.x]);
	//		pixel_color += rayColor(r, d_world, &sharedRandState[threadIdx.y][threadIdx.x]);
	//	}
	//	writeColor(data, index, camera->m_PixelSampleScale * pixel_color);

	//}
	__global__ void render(unsigned char* data, Camera* camera, HittableList* d_world, curandState* rand_state, int imageWidth, int imageHeight)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x >= imageWidth && y >= imageHeight) return;

		int pixel_index = y * imageWidth + x;

		curandState local_rand_state = rand_state[pixel_index];

		int data_pixel_index = (y * imageWidth + x) * 4;

		Color pixel_color(0, 0, 0);
		for (int sample = 0; sample < camera->m_SamplesPerPixel; sample++) {
			const Ray& r = camera->getRay(x, y, imageWidth, imageHeight, &local_rand_state);
			pixel_color += rayColor(r, d_world, &local_rand_state);
		}

		rand_state[pixel_index] = local_rand_state;

		writeColor(data, data_pixel_index, camera->m_PixelSampleScale * pixel_color);
	}
}
int main()
{
	const float ASPECT_RATIO = 16.0f / 9.0f;
	int WIDTH = 1440;
	int HEIGHT = static_cast<int>(WIDTH / ASPECT_RATIO);
	HEIGHT = (HEIGHT < 1) ? 1 : HEIGHT;

	CRT::Vec3 cameraPosition(0, 3, 0);
	CRT::Vec3 cameraTarget(0, 0, -1);
	CRT::Vec3 worldY(0, 1, 0);
	float aperture = 0.0001f;
	float cameraSpeed = 0.1f;

	CRT::Camera h_camera(ASPECT_RATIO, 120.0f, cameraPosition, cameraTarget, worldY, aperture, float((cameraPosition - cameraTarget).length()));

	CRT::Camera* d_camera;
	checkCudaErrors(cudaMalloc(&d_camera, sizeof(CRT::Camera)));

	checkCudaErrors(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));

	const int colorChannels = 4;
	const size_t dataImageSize = WIDTH * HEIGHT * colorChannels;

	unsigned char* d_imageData;
	checkCudaErrors(cudaMallocManaged(&d_imageData, dataImageSize));

	dim3 threads(16, 16);
	dim3 blocks((WIDTH + threads.x - 1) / threads.x, (HEIGHT + threads.y - 1) / threads.y);
	int threadsPerBlock = threads.x * threads.y;
	int totalBlocks = blocks.x * blocks.y;
	int totalThreads = totalBlocks * threadsPerBlock;

	//CRT::Hittable** d_sphere;
	//CRT::Material** d_materials;
	//checkCudaErrors(cudaMalloc(&d_sphere, 4 * sizeof(CRT::Hittable*)));
	//checkCudaErrors(cudaMalloc(&d_materials, 3 * sizeof(CRT::Material*)));
	//CRT::initWorld CUDA_KERNEL(1, 1)(d_world, d_sphere, d_camera, d_materials);
	//checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());

// Call the global function to create the random world

	// Don't forget to synchronize after the kernel launch
	cudaDeviceSynchronize();

	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc(&d_rand_state, totalThreads * sizeof(curandState)));
	unsigned long long seed = rand();
	CRT::initRandState CUDA_KERNEL(blocks, threads)(d_rand_state, seed, WIDTH, HEIGHT);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	CRT::HittableList* d_world;
	checkCudaErrors(cudaMalloc(&d_world, sizeof(CRT::HittableList)));
	checkCudaErrors(cudaMemset(d_world, 0, sizeof(CRT::HittableList)));  // Initialize the world to zeros
	CRT::createRandomWorld CUDA_KERNEL(1, 1)(d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	CRT::render CUDA_KERNEL(blocks, threads)(d_imageData, d_camera, d_world, d_rand_state, WIDTH, HEIGHT);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "CUDA Raytracer Output");
	sf::Texture texture;
	texture.create(WIDTH, HEIGHT);
	sf::Sprite sprite(texture);

	sf::Image image;
	image.create(WIDTH, HEIGHT, d_imageData);
	image.flipVertically();
	texture.update(image);

	sf::Clock clock;
	const float targetFPS = 60.0f;
	const sf::Time targetFrameTime = sf::seconds(1.0f / targetFPS);

	window.setMouseCursorVisible(true);
	window.setMouseCursorGrabbed(false);

	bool isRightMousePressed = false;

	while (window.isOpen())
	{
		sf::Time elapsedTime = clock.getElapsedTime();
		float deltaTime = elapsedTime.asSeconds();
		clock.restart();

		// Handle events
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
			//else if (event.type == sf::Event::Resized)

			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Right)
				{
					isRightMousePressed = true;
					window.setMouseCursorVisible(false);
					//window.setMouseCursorGrabbed(true);
				}
			}
			else if (event.type == sf::Event::MouseButtonReleased)
			{
				if (event.mouseButton.button == sf::Mouse::Right)
				{
					isRightMousePressed = false;
					window.setMouseCursorVisible(true);
					//window.setMouseCursorGrabbed(false);
					sf::Mouse::setPosition({ WIDTH / 2,HEIGHT / 2 }, window);
				}
			}
			else if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::PageUp)
				{
					h_camera.adjustFocusDistance(0.1f);
					std::cout << "Focus Distance: " << h_camera.getFocusDistance() << std::endl;
				}
				else if (event.key.code == sf::Keyboard::PageDown)
				{
					h_camera.adjustFocusDistance(-0.1f);
					std::cout << "Focus Distance: " << h_camera.getFocusDistance() << std::endl;
				}
			}
		}

		// Update camera
		sf::Vector2i mousePos = sf::Mouse::getPosition(window);
		h_camera.updateCamera(deltaTime, WIDTH, HEIGHT,
			static_cast<float>(mousePos.x), static_cast<float>(mousePos.y),
			isRightMousePressed);

		// Update camera on GPU
		checkCudaErrors(cudaMemcpy(d_camera, &h_camera, sizeof(CRT::Camera), cudaMemcpyHostToDevice));

		// Re-render the scene
		CRT::render CUDA_KERNEL(blocks, threads)(d_imageData, d_camera, d_world, d_rand_state, WIDTH, HEIGHT);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		// Update the texture and draw
		sf::Image image;
		image.create(WIDTH, HEIGHT, d_imageData);
		image.flipVertically();
		texture.update(image);

		window.clear();
		window.draw(sprite);
		window.display();
	}

	// Clean up (after the loop ends)
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_imageData));

	return 0;
}