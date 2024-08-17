#pragma once
#include <SFML/Graphics.hpp>
#include "Core/Camera.cuh"

class WindowManager {
public:
	struct WindowConfig {
		sf::RenderWindow window;
		sf::Texture texture;
		sf::Sprite sprite;

		WindowConfig(int width, int height)
			: window(sf::VideoMode(width, height), "CUDA Raytracer Output") {
			if (!texture.create(width, height)) {
				throw std::runtime_error("Failed to create SFML texture");
			}
			sprite.setTexture(texture);
		}

		WindowConfig(const WindowConfig&) = delete;
		WindowConfig& operator=(const WindowConfig&) = delete;
		WindowConfig(WindowConfig&&) = default;
		WindowConfig& operator=(WindowConfig&&) = default;
	};

	WindowManager(int width, int height);
	WindowConfig& getWindowConfig();
	void handleEvents(bool& isRightMousePressed, CRT::Camera& camera);
	void drawFrame(unsigned char* dImageData, int width, int height);

private:
	WindowConfig m_WindowConfig;
	sf::Image m_Image;
	int m_CurrentWidth;
	int m_CurrentHeight;
};

// Implementation
WindowManager::WindowManager(int width, int height)
	: m_WindowConfig(width, height), m_CurrentWidth(width), m_CurrentHeight(height)
{
	m_Image.create(width, height);
}

WindowManager::WindowConfig& WindowManager::getWindowConfig() {
	return m_WindowConfig;
}

void WindowManager::handleEvents(bool& isRightMousePressed, CRT::Camera& camera) {
	sf::Event event;
	while (m_WindowConfig.window.pollEvent(event)) {
		if (event.type == sf::Event::Closed)
			m_WindowConfig.window.close();
		else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Right) {
			isRightMousePressed = true;
			m_WindowConfig.window.setMouseCursorVisible(false);
		}
		else if (event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Right) {
			isRightMousePressed = false;
			m_WindowConfig.window.setMouseCursorVisible(true);
			sf::Mouse::setPosition(sf::Vector2i(m_WindowConfig.window.getSize().x / 2, m_WindowConfig.window.getSize().y / 2), m_WindowConfig.window);
		}
		else if (event.type == sf::Event::KeyPressed) {
			if (event.key.code == sf::Keyboard::PageUp)
				camera.adjustFocusDistance(0.1f);
			else if (event.key.code == sf::Keyboard::PageDown)
				camera.adjustFocusDistance(-0.1f);
		}
		else if (event.type == sf::Event::Resized) {
			// Update image size if window is resized
			m_CurrentWidth = event.size.width;
			m_CurrentHeight = event.size.height;
			m_Image.create(m_CurrentWidth, m_CurrentHeight);
			m_WindowConfig.texture.create(m_CurrentWidth, m_CurrentHeight);
		}
	}
}

void WindowManager::drawFrame(unsigned char* dImageData, int width, int height) {
	// Check if the image size needs to be updated
	if (width != m_CurrentWidth || height != m_CurrentHeight) {
		m_CurrentWidth = width;
		m_CurrentHeight = height;
		m_Image.create(width, height);
		m_WindowConfig.texture.create(width, height);
	}
	CUDA_CHECK(cudaMemcpy((void*)m_Image.getPixelsPtr(), dImageData, width * height * 4, cudaMemcpyDeviceToHost));
	m_Image.flipVertically();
	m_WindowConfig.texture.update(m_Image);
	m_WindowConfig.window.clear();
	m_WindowConfig.window.draw(m_WindowConfig.sprite);
	m_WindowConfig.window.display();
}
