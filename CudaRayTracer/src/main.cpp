#include <SFML/Graphics.hpp>
#include<iostream>
#include<thread>

#include"stb_image_write.h"
#include"Core/Utility.h"
#include"Core/Sphere.h"
#include"Core/HittableList.h"
#include"Core/Camera.h"
#include"Core/TimeHelper.h"

namespace BRT
{



	// Scre
	const float ASPECT_RATIO = 16.0f / 9.0f;
	const int WIDTH = 1000;
	const int HEIGHT = static_cast<int>(WIDTH / ASPECT_RATIO);
	const int SAMPLES_PER_PIXEL = 30;
	const int MAX_DEPTH = 10;


	// Camera and Viewport

	float viewport_height = 2.0;
	float viewport_width = ASPECT_RATIO * viewport_height;
	float focal_length = 1.0;

	glm::vec3 CameraOrigin{ 0, 0, 0 };
	glm::vec3 VP_Horizontal{ viewport_width, 0, 0 };
	glm::vec3 VP_Vertical{ 0, viewport_height, 0 };

	glm::vec3 ViewportBottomLeftCorner = CameraOrigin - VP_Horizontal / 2.f - VP_Vertical / 2.f - glm::vec3(0, 0, focal_length);



	glm::vec3 RayColor(const Ray& ray, const HittableList& world, int rayDepth)
	{
		if (rayDepth <= 0)
			return { 0, 0, 0 };

		HitInfo info;
		if (world.Hit(ray, 0.001, Utility::s_Infinity, info))
		{

			Ray scattered{};
			glm::vec3 attenuation{};
			if (info.Material->Scatter(ray, info, attenuation, scattered))
			{
				return attenuation * RayColor(scattered, world, rayDepth - 1);
			}

			//glm::vec3 target = info.Point  + RandomPointInHemisphere(info.Normal);
			//return  0.5f * RayColor({ info.Point, target - info.Point }, world, rayDepth - 1);
			//return 0.5f * (info.Normal + glm::vec3(1, 1, 1));
		}

		glm::vec3 unitDirection = glm::normalize(ray.GetDirection());
		float t = 0.5 * (unitDirection.y + 1.0);
		return float(1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0);
	}

	void WriteColor(unsigned char* data, uint32_t index, glm::vec3& color)
	{

		// Divide the color by the number of samples and gamma correction.
		auto scale = (1.0 / SAMPLES_PER_PIXEL);
		color.r *= scale;
		color.g *= scale;
		color.b *= scale;

		const glm::vec3 gamma(1.f / 2.2f);

		color = glm::pow(color, gamma);

		data[index] = static_cast<char>(255 * glm::clamp(color.r, 0.0f, 0.999f));         // Red channel
		data[index + 1] = static_cast<char>(255 * glm::clamp(color.g, 0.0f, 0.999f));    // Green channel
		data[index + 2] = static_cast<char>(255 * glm::clamp(color.b, 0.0f, 0.999f));    // Blue channel
		data[index + 3] = 255;

	}

	const int numThreads = std::thread::hardware_concurrency() ;
	std::vector<std::atomic<bool>> threadStatus(numThreads);

	void RenderImagePart(int threadIndex, int start, int end, const BRT::Camera& camera, const BRT::HittableList& world, uint8_t* data) {
		for (int i = start; i < end; i += 4)
		{
			int x = (i % (BRT::WIDTH * 4) / 4);
			int y = (i / (BRT::WIDTH * 4));

			glm::vec3 pixelColor(0, 0, 0);
			for (int s = 0; s < BRT::SAMPLES_PER_PIXEL; ++s) {
				auto u = (x + BRT::Utility::RandomDouble()) / (BRT::WIDTH - 1);
				auto v = (y + BRT::Utility::RandomDouble()) / (BRT::HEIGHT - 1);

				BRT::Ray ray = camera.GetRay(u, v);
				pixelColor += BRT::RayColor(ray, world, BRT::MAX_DEPTH);
			}

			BRT::WriteColor(data, i, pixelColor);

		}
		threadStatus[threadIndex].store(false, std::memory_order_relaxed);
	}



}

BRT::HittableList RandomScene();

int main()
{

	BRT::HittableList world = RandomScene();


	/*   auto material_ground =  BRT::CreateShared<BRT::Matte>(glm::vec3(0.2, 0.2, 0.1));
	   auto material_center =  BRT::CreateShared<BRT::Dialectric>(1.5f);
	   auto material_left =    BRT::CreateShared<BRT::Metal>(glm::vec3(0.8, 0.8, 0.8), 0.7f);
	   auto material_right =   BRT::CreateShared<BRT::Matte>(glm::vec3(0.8, 0.9, 0.2));

	   world.AddObject (BRT::CreateShared<BRT::Sphere>(glm::vec3(0.0, -100.5, -1.0), 100.0, material_ground));
	   world.AddObject (BRT::CreateShared<BRT::Sphere>(glm::vec3(0.0, 0.0, -1.0), 0.3, material_center));
	   world.AddObject (BRT::CreateShared<BRT::Sphere>(glm::vec3(-1.0, 0.0, -1.0), -0.45, material_left));
	   world.AddObject(BRT::CreateShared<BRT::Sphere>(glm::vec3(-1.0, 0.0, -1.0), 0.5, material_left));
	   world.AddObject (BRT::CreateShared<BRT::Sphere>(glm::vec3(1.0, 0.0, -1.0), 0.5, material_right));*/



	glm::vec3 position(23, 7, 1);
	glm::vec3 target(0, 0, 0);
	glm::vec3 worldY(0, 1, 0);
	float aperture = 0.1f;


	BRT::Camera camera{ BRT::ASPECT_RATIO, 20.0, 2.f, position, target, worldY, aperture, glm::length(position - target) };
	// Render
	std::cout << "Resolution:" << BRT::WIDTH << " X " << BRT::HEIGHT << "\n";

	//unsigned char* data = new unsigned char[BRT::HEIGHT * BRT::WIDTH * 4];
	uint8_t* data = new uint8_t[BRT::HEIGHT * BRT::WIDTH * 4];
	//std::cerr << "\rScanlines remaining: " << y << ' ' << std::flush;
   // Generate the gradient

	sf::RenderWindow window(sf::VideoMode(BRT::WIDTH, BRT::HEIGHT), "Raytracer Output");
	sf::Texture texture;
	texture.create(BRT::WIDTH, BRT::HEIGHT);
	sf::Sprite sprite;
	sprite.setTexture(texture);

	BRT::Timer timer;
	timer.Start();

	int totalPixels = BRT::HEIGHT * BRT::WIDTH * 4;

	// Calculate the number of pixels each thread will handle.
	int pixelsPerThread = totalPixels / BRT::numThreads;

	// Create a vector of threads.
	std::vector<std::thread> threads;
	for (int t = 0; t < BRT::numThreads; ++t) {
		// Calculate the start and end indices for each thread.
		int start = t * pixelsPerThread;
		int end = (t == BRT::numThreads - 1) ? totalPixels : start + pixelsPerThread;
	
		BRT::threadStatus[t].store(true, std::memory_order_relaxed); // Initialize all flags to true
		// Launch a thread to render a portion of the image.
		threads.emplace_back(BRT::RenderImagePart, t, start, end, std::ref(camera), std::ref(world), data);
	}


	sf::Image image;

	// Check the status of each thread
	while (true)
	{
		bool allThreadsCompleted = true;

		for (int i = 0; i < BRT::numThreads; ++i) {
			if (BRT::threadStatus[i].load(std::memory_order_relaxed)) {
				allThreadsCompleted = false;
				break;
			}
		}
		if (allThreadsCompleted) {
			break; // All threads have completed
		}
		image.create(BRT::WIDTH, BRT::HEIGHT, data);
		image.flipVertically();

		texture.update(image);

		window.clear();

		window.draw(sprite);

		window.display();
	}


	//Wait for all threads to finish.
	for (auto& thread : threads)
	{
		thread.join();
	}

	timer.Stop();



	std::cerr << "\nRendering complete!\n";

	std::cout << "Time:" << timer.GetTimeMilliseconds() << " ms";

	stbi_flip_vertically_on_write(true);

		int result = stbi_write_jpg("outputs/jpg_test_.jpg", BRT::WIDTH, BRT::HEIGHT, 4, data, 100);

	if (!result) {
		std::cout << "WRITE ERROR!!";
	}

	delete[] data;


	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed) {
				window.close();
			}
		}

	}



	return 0;
}


BRT::HittableList RandomScene()
{
	BRT::HittableList world;

	auto ground_material = BRT::CreateShared<BRT::Matte>(glm::vec3(0.25, 0.5, 0.35));
	world.AddObject(BRT::CreateShared<BRT::Sphere>(glm::vec3(0, -1000, 0), 1000, ground_material));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = BRT::Utility::RandomDouble();
			glm::vec3 center(a + 0.9 * BRT::Utility::RandomDouble(), 0.2, b + 0.9 * BRT::Utility::RandomDouble());

			if ((center - glm::vec3(4, 0.2, 0)).length() > 0.9) {
				BRT::Shared<BRT::Material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = BRT::Utility::RandomVector(0.f, 1.f) * BRT::Utility::RandomVector(0.f, 1.f);
					sphere_material = BRT::CreateShared<BRT::Matte>(albedo);
					world.AddObject(BRT::CreateShared<BRT::Sphere>(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = BRT::Utility::RandomVector(0.f, 1.f);
					auto fuzz = BRT::Utility::RandomDouble(0, 0.8);
					sphere_material = BRT::CreateShared<BRT::Metal>(albedo, fuzz);
					world.AddObject(BRT::CreateShared<BRT::Sphere>(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = BRT::CreateShared<BRT::Dialectric>(1.5);
					world.AddObject(BRT::CreateShared<BRT::Sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = BRT::CreateShared<BRT::Dialectric>(1.5);
	world.AddObject(BRT::CreateShared<BRT::Sphere>(glm::vec3(0, 1, 0), 1.0, material1));

	auto material2 = BRT::CreateShared<BRT::Matte>(glm::vec3(0.4, 0.2, 0.1));
	world.AddObject(BRT::CreateShared<BRT::Sphere>(glm::vec3(-4, 1, 0), 1.0, material2));

	auto material3 = BRT::CreateShared<BRT::Metal>(glm::vec3(0.7, 0.6, 0.5), 0.0);
	world.AddObject(BRT::CreateShared<BRT::Sphere>(glm::vec3(4, 1, 0), 1.0, material3));

	return world;
}