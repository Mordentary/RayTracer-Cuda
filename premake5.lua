workspace "CudaRayTracer"
	architecture "x86_64"
	startproject "CudaRayTracer"

	configurations
	{
		"Debug",
		"Release"
	}
	flags
	{
		"MultiProcessorCompile"
	}

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
IncludeDir = {}
IncludeDir["glm"] = "CudaRayTracer/vendor/glm"
IncludeDir["stb_image"] = "CudaRayTracer/vendor/stb_image"
IncludeDir["SFML"] = "CudaRayTracer/vendor/SFML-2.6.0/include"



project "CudaRayTracer"
    location "CudaRayTracer"
	kind "ConsoleApp"
	language "C++"
	cppdialect "C++17"
	staticruntime "on"


	targetdir("bin/" .. outputdir ..  "/%{prj.name}")
	objdir("bin-int/" .. outputdir ..  "/%{prj.name}")


	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp",
		"%{prj.name}/vendor/glm/glm/**.hpp",
		"%{prj.name}/vendor/glm/glm/**.inl",
		"%{prj.name}/vendor/stb_image/**.h",
		"%{prj.name}/vendor/stb_image/**.cpp",
		"%{prj.name}/vendor/SFML-2.6.00/include/**.hpp"
	}
	
	libdirs
    {
		"%{prj.name}/vendor/SFML-2.6.0/lib"
    }
    
    links
    {
		"sfml-graphics.lib", 
		"sfml-window.lib",
		"sfml-system.lib"
    }

	defines
	{	
		"_CRT_SECURE_NO_WARNINGS" 
	}

	includedirs
	{
		"%{prj.name}/src",
		"%{IncludeDir.glm}",
		"%{IncludeDir.stb_image}",
		"%{IncludeDir.SFML}"

	}



	filter "system:windows"
		systemversion "latest"

	filter "configurations:Debug"
		defines "BRT_DEBUG"
		runtime "Debug"
		symbols "on"
		
	filter "configurations:Release"
		defines "BRT_REALESE"
		runtime "Release"
		symbols "Off"
	
	filter {}  

