#include "Vec2.cuh"

namespace geometry
{
    __host__ __device__ Vec2::Vec2(float v0, float v1) : values{v0, v1} {}

    __host__ __device__ float Vec2::x() const { return values[0]; }

    __host__ __device__ float &Vec2::x() { return values[0]; }

    __host__ __device__ float Vec2::y() const { return values[1]; }

    __host__ __device__ float &Vec2::y() { return values[1]; }
} // namespace geometry
