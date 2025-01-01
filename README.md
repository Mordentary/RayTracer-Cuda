### CUDA-Accelerated Ray Tracer

This project is an enhanced version of my basic ray tracer, now accelerated using CUDA. It leverages GPU parallelism to significantly improve rendering performance.

#### Key Features:
- **Full CUDA Implementation**: The entire ray-tracing pipeline is GPU-based.
- **BVH Acceleration on GPU**: Uses a Bounding Volume Hierarchy to speed up ray-scene intersections, especially for large models.
- **OBJ Model Loading**
- **Materials**: Supports Diffuse, Metal, Dielectric, and DiffuseLight materials.

Rendering in 2K, with 2000 samples per pixel and a maximum of 20 bounces, takes several minutes on my 3050 Ti. Lower sample rates work in real time.
![](https://github.com/Mordentary/RayTracer-Cuda/blob/master/Screenshots/Cornell-box-with-bunny.jpg?raw=true)
