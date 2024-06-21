#pragma once
#include <memory>
#include <glm/glm.hpp>

namespace BRT
{
	template<typename T>
	using Shared = std::shared_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Shared<T> CreateShared(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}

	template<typename T>
	using Scoped = std::unique_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Scoped<T> CreateScoped(Args&& ... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}

}