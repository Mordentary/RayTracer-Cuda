#pragma once

#include "CUDAHelpers.h"
#include "SceneManager.h"
#include "WindowManager.h"
#include "Core/Camera.cuh"
#include "CUDARenderer.cuh"

class Raytracer {
public:
	Raytracer(int width, int height, float aspectRatio, float verticalFOV, float aperture);
	~Raytracer();

	void run();

private:
	void initializeScene();
	void handleEvents(bool& isRightMousePressed);
	void updateAndRender(float deltaTime, bool isRightMousePressed);
	void drawFrame();

	int m_Width;
	int m_Height;
	float m_AspectRatio;
	float m_VerticalFOV;
	float m_Aperture;
	CRT::Camera m_Camera;
	CUDAHelpers::RenderConfig m_RenderConfig;

	SceneManager m_SceneManager;
	CUDARenderer m_Renderer;
	WindowManager m_WindowManager;

};

// Implementation

Raytracer::Raytracer(int width, int height, float aspectRatio, float verticalFOV, float aperture)
	: m_Width(width), m_Height(height), m_AspectRatio(aspectRatio),
	m_VerticalFOV(verticalFOV), m_Aperture(aperture),
	m_SceneManager(width, height),
	m_WindowManager(width, height),
	m_Renderer(width, height) 
{
	initializeScene();
}

Raytracer::~Raytracer() {
	// Cleanup is now handled by CUDARenderer
}

void Raytracer::run() {
	WindowManager::WindowConfig& windowConfig = m_WindowManager.getWindowConfig();
	sf::Clock clock;
	const float targetFPS = 60.0f;
	const sf::Time targetFrameTime = sf::seconds(1.0f / targetFPS);

	bool isRightMousePressed = false;

	while (windowConfig.window.isOpen()) {
		sf::Time elapsedTime = clock.getElapsedTime();
		float deltaTime = elapsedTime.asSeconds();
		clock.restart();

		handleEvents(isRightMousePressed);
		updateAndRender(deltaTime, isRightMousePressed);
		drawFrame();

		sf::sleep(targetFrameTime - clock.getElapsedTime());
	}
}

void Raytracer::handleEvents(bool& isRightMousePressed) {
	m_WindowManager.handleEvents(isRightMousePressed, m_Camera);
}

void Raytracer::initializeScene() {
	// Set up camera
	CRT::Vec3 cameraPosition(0, 4, 4);
	CRT::Vec3 cameraTarget(0, 0, 0);
	CRT::Vec3 worldY(0, 1, 0);
	float focusDist = (cameraPosition - cameraTarget).length();

	m_Camera = CRT::Camera(m_AspectRatio, m_VerticalFOV, cameraPosition, cameraTarget, worldY, m_Aperture, focusDist);

	// Set up render configuration
	m_RenderConfig = CUDAHelpers::createRenderConfig(m_Width, m_Height);

	// Initialize CUDA resources
	m_Renderer.initialize(m_RenderConfig);
	m_SceneManager.initializeScene(m_RenderConfig, m_Renderer.getRandState());
}

void Raytracer::updateAndRender(float deltaTime, bool isRightMousePressed) {
	WindowManager::WindowConfig& windowConfig = m_WindowManager.getWindowConfig();
	sf::Vector2i mousePos = sf::Mouse::getPosition(windowConfig.window);

	m_Camera.updateCamera(deltaTime, m_Width, m_Height, static_cast<float>(mousePos.x), static_cast<float>(mousePos.y), isRightMousePressed);

	m_Renderer.updateCamera(m_Camera);
	m_Renderer.render(m_SceneManager.getBVHNodes(), m_SceneManager.getWorld());
}

void Raytracer::drawFrame() {
	m_WindowManager.drawFrame(m_Renderer.getImageData(), m_Width, m_Height);
}
