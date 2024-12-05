#ifndef VEC2_CUH
#define VEC2_CUH

namespace geometry
{
    class Vec2
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Vec2(float v0 = 0.0f, float v1 = 0.0f);

        __host__ __device__ float x() const;

        __host__ __device__ float &x();

        __host__ __device__ float y() const;

        __host__ __device__ float &y();

        __host__ __device__ float operator[](uint index) const { return values[index]; }

        __host__ __device__ float &operator[](uint index) { return values[index]; }

        // ----------------------------------------------------------------
        // --- Public attributes
        float values[2];

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
    };

    __host__ __device__ inline float dot(const Vec2 &u, const Vec2 &v)
    {
        return u.x() * v.x() + u.y() * v.y();
    }

    __host__ __device__ inline Vec2 operator+(const Vec2 &u, const Vec2 &v)
    {
        return Vec2(u.x() + v.x(), u.y() + v.y());
    }

    __host__ __device__ inline Vec2 operator-(const Vec2 &u, const Vec2 &v)
    {
        return Vec2(u.x() - v.x(), u.y() - v.y());
    }

    using Point2 = Vec2;
} // namespace geometry

#endif // VEC2_CUH