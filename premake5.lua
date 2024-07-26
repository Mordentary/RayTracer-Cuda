workspace "CudaRayTracer"
    architecture "x86_64"
    startproject "CudaRayTracer"
    configurations { "Debug", "Release" }
    flags { "MultiProcessorCompile" }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

IncludeDir = {}
IncludeDir["glm"] = "CudaRayTracer/vendor/glm"
IncludeDir["stb_image"] = "CudaRayTracer/vendor/stb_image"
IncludeDir["SFML"] = "CudaRayTracer/vendor/SFML-2.6.0/include"

-- CUDA configuration
cudaToolkitDir = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5" 
cudaToolkitIncludeDir = cudaToolkitDir .. "/include"
cudaToolkitLibDir = cudaToolkitDir .. "/lib/x64"

project "CudaRayTracer"
    location "CudaRayTracer"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "on"


    targetdir("bin/" .. outputdir ..  "/%{prj.name}")
    objdir("bin-int/" .. outputdir ..  "/%{prj.name}")

    files {
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.cu",
        "%{prj.name}/src/**.cuh",
        "%{prj.name}/vendor/stb_image/**.h",
        "%{prj.name}/vendor/stb_image/**.cpp",
        "%{prj.name}/vendor/SFML-2.6.00/include/**.hpp"
    }

    libdirs {
        "%{prj.name}/vendor/SFML-2.6.0/lib",
        cudaToolkitLibDir
    }

    links {
        "sfml-graphics.lib", 
        "sfml-window.lib",
        "sfml-system.lib",
        "cudart.lib",
        "cuda.lib",
        "cublas.lib",
        "cufft.lib"
    }

    defines { "_CRT_SECURE_NO_WARNINGS" }

    includedirs {
        "%{prj.name}/src",
        "%{IncludeDir.glm}",
        "%{IncludeDir.stb_image}",
        "%{IncludeDir.SFML}",
        cudaToolkitIncludeDir
    }


    filter "system:windows"
        systemversion "latest"

    filter "configurations:Debug"
        defines "CRT_DEBUG"
        runtime "Debug"
        symbols "on"

    filter "configurations:Release"
        defines "CRT_RELEASE"
        runtime "Release"
        optimize "on"

    filter {}