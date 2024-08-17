#pragma once

#include "Vec3.cuh"
#include "Ray.cuh"
#include "Utility.cuh"
#include <cuda_runtime.h>
#include <SFML/Window/Keyboard.hpp>

namespace CRT
{
	__device__ static constexpr int DEFAULT_SAMPLES_PER_PIXEL = 1;

	class Camera
	{
	public:
		__host__ __device__ Camera() = default;


		__host__ __device__ Camera(float aspectRatio, float fov, Vec3 position, Vec3 target, Vec3 up, float aperture, float focusDist)
			: m_AspectRatio(aspectRatio), m_VerticalFOV(fov), m_Position(position), m_Aperture(aperture), m_FocusDist(focusDist)
		{
			m_WorldUp = up;
			m_SamplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;
			m_PixelSampleScale = 1.0f / m_SamplesPerPixel;
			m_Yaw = -90.0f;
			m_Pitch = 0.0f;
			m_MovementSpeed = 2.5f;
			m_MouseSensitivity = 0.3f;
			m_HighQualityMode = m_CameraMoves = m_CameraRotates = false;
			updateCameraVectors();
		}

		__device__ Ray getRay(int pixel_x, int pixel_y, int image_width, int image_height, curandState* rand_state) const
		{
			Vec3 rd = m_LensRadius * Utility::randomPointInUnitDisk(rand_state);
			Vec3 offset = m_Right * rd.x() + m_Up * rd.y();

			float u = (float(pixel_x) + Utility::randomFloat(rand_state)) / float(image_width);
			float v = (float(pixel_y) + Utility::randomFloat(rand_state)) / float(image_height);

			return Ray(
				m_Position + offset,
				m_LowerLeftCorner + u * m_Horizontal + v * m_Vertical - m_Position - offset
			);
		}

		__host__ void updateCamera(float deltaTime, int windowWidth, int windowHeight, float mouseX, float mouseY, bool mousePressed)
		{
			updateRotation(deltaTime, windowWidth, windowHeight, mouseX, mouseY, mousePressed);
			updatePosition(deltaTime);
			updateCameraVectors();

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::F))
			{
				m_HighQualityMode = !m_HighQualityMode;
			}

			if (m_CameraRotates || m_CameraMoves)
			{
				m_SamplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;
				m_HighQualityMode = false;
			}
			else if (m_HighQualityMode)
			{
				m_SamplesPerPixel = 500;
			}
			else
			{
				m_SamplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL;
			}

			m_PixelSampleScale = 1.f / m_SamplesPerPixel;
		}

		__host__ bool isCameraInMotion()
		{
			return m_CameraRotates || m_CameraMoves;
		}

		__host__ void adjustFocusDistance(float delta)
		{
			m_FocusDist = fmaxf(0.1f, m_FocusDist + delta);
			updateCameraVectors();
		}

		__host__ float getFocusDistance() const { return m_FocusDist; }

	private:
		__host__ void updateRotation(float deltaTime, int windowWidth, int windowHeight, float mouseX, float mouseY, bool mousePressed)
		{
			static bool firstMouse = true;
			static float lastX = windowWidth / 2.0f;
			static float lastY = windowHeight / 2.0f;
			static float smoothX = lastX;
			static float smoothY = lastY;
			static const float smoothFactor = 0.5f; // Adjust this value to change smoothing intensity

			if (mousePressed)
			{
				if (firstMouse)
				{
					lastX = mouseX;
					lastY = mouseY;
					smoothX = mouseX;
					smoothY = mouseY;
					firstMouse = false;
					return; // Skip the first frame to avoid a large initial jump
				}

				m_CameraRotates = true;

				// Smooth out the mouse movement
				smoothX = smoothX * (1 - smoothFactor) + mouseX * smoothFactor;
				smoothY = smoothY * (1 - smoothFactor) + mouseY * smoothFactor;

				float xoffset = smoothX - lastX;
				float yoffset = smoothY - lastY; // Reversed: y ranges bottom to top

				lastX = smoothX;
				lastY = smoothY;

				xoffset *= -m_MouseSensitivity; // Inverted X rotation
				yoffset *= -m_MouseSensitivity; // Inverted Y rotation

				m_Yaw += xoffset;
				m_Pitch += yoffset;

				m_Pitch = fmaxf(-89.0f, fminf(89.0f, m_Pitch)); // Clamp pitch
			}
			else
			{
				firstMouse = true; // Reset when mouse is released
				m_CameraRotates = false;
			}
		}
		__host__ void updatePosition(float deltaTime)
		{
			float velocity = m_MovementSpeed * deltaTime;

			Vec3 prevPosition = m_Position;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
				m_Position -= m_Front * velocity;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
				m_Position += m_Front * velocity;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
				m_Position -= m_Right * velocity;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
				m_Position += m_Right * velocity;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
				m_Position += m_WorldUp * velocity;
			if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl))
				m_Position -= m_WorldUp * velocity;

			if (m_Position != prevPosition)
				m_CameraMoves = true;
			else
				m_CameraMoves = false;
		}

		__host__ __device__ void updateCameraVectors()
		{
			Vec3 front;
			front.e[0] = -cosf(m_Yaw * PI / 180.0f) * cosf(m_Pitch * PI / 180.0f);
			front.e[1] = -sinf(m_Pitch * PI / 180.0f);
			front.e[2] = -sinf(m_Yaw * PI / 180.0f) * cosf(m_Pitch * PI / 180.0f);
			m_Front = unitVector(front);

			// Re-calculate the Right and Up vector
			m_Right = unitVector(cross(m_Front, m_WorldUp));
			m_Up = unitVector(cross(m_Right, m_Front));

			// Update viewport
			float theta = m_VerticalFOV * PI / 180.0f;
			float h = tanf(theta / 2.0f);
			float viewportHeight = 2.0f * h;
			float viewportWidth = m_AspectRatio * viewportHeight;

			m_Horizontal = m_FocusDist * viewportWidth * m_Right;
			m_Vertical = m_FocusDist * viewportHeight * m_Up;
			m_LowerLeftCorner = m_Position - m_Horizontal / 2.0f - m_Vertical / 2.0f - m_FocusDist * m_Front;

			m_LensRadius = m_Aperture / 2.0f;
		}

	public:
		int m_SamplesPerPixel;
		float m_PixelSampleScale;
	private:
		Vec3 m_Position;
		Vec3 m_Front;
		Vec3 m_Up;
		Vec3 m_Right;
		Vec3 m_WorldUp;

		float m_Yaw;
		float m_Pitch;
		float m_MovementSpeed;
		float m_MouseSensitivity;

		float m_AspectRatio;
		float m_VerticalFOV;
		float m_Aperture;
		float m_FocusDist;

		Vec3 m_LowerLeftCorner;
		Vec3 m_Horizontal;
		Vec3 m_Vertical;
		float m_LensRadius;

		bool m_CameraMoves, m_CameraRotates;
		bool m_HighQualityMode;
	};

	__constant__ Camera d_camera;
}