#ifndef PERLINNOISEGENERATOR_CUH
#define PERLINNOISEGENERATOR_CUH

#include "PPMImage.cuh"
#include <stdint.h>

namespace geometry
{
    class Vec2
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ Vec2(float v0 = 0.0f, float v1 = 0.0f) : values{v0, v1} {}

        __host__ __device__ float x() const { return values[0]; }

        __host__ __device__ float &x() { return values[0]; }

        __host__ __device__ float y() const { return values[1]; }

        __host__ __device__ float &y() { return values[1]; }

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

namespace procedural_generation
{
    class PerlinNoiseGenerator
    {
        /*
        Based on "C++: Perlin Noise Tutorial" by Zipped @ https://www.youtube.com/watch?v=kCIaHqb60Cw
        */
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static constexpr uint GRID_ROWS = 600;
        static constexpr uint GRID_COLUMNS = 600;

        // ----------------------------------------------------------------
        // --- Public methods

        __host__ PerlinNoiseGenerator(uint userSeed = 0)
        {
            seed = userSeed != 0 ? userSeed : (uint)time(nullptr);
        }

        __host__ static void generateAndSave(uint octaveCount = 12, const std::string &outputName = "noise")
        {
            procedural_generation::PerlinNoiseGenerator generator;
            graphics::PPMImage blackAndWhiteImage(GRID_ROWS, GRID_COLUMNS, 1.0f);
            graphics::PPMImage terrainLikeImage(GRID_ROWS, GRID_COLUMNS, 1.0f);
            generator.generate(octaveCount);

            std::string blackAndWhitePath = outputName + ".ppm";
            std::string terrainLikePath = outputName + "_terrain.ppm";

            for (size_t i = 0; i < GRID_ROWS; i++)
            {
                for (size_t j = 0; j < GRID_COLUMNS; j++)
                {
                    graphics::ColorChannel intensity = generator.grid[i][j];

                    graphics::RGBPixel pixel(intensity, intensity, intensity);
                    blackAndWhiteImage.setPixel(i, j, pixel);

                    pixel = generator.getTerrainColor(intensity);
                    terrainLikeImage.setPixel(i, j, pixel);
                }
            }

            blackAndWhiteImage.save(blackAndWhitePath);
            terrainLikeImage.save(terrainLikePath);
        }

        __host__ __device__ void generate(uint octaveCount)
        {
            constexpr uint BASE_FREQUENCY = (GRID_ROWS * GRID_COLUMNS) / 2592;
            for (size_t i = 0; i < GRID_ROWS; i++)
            {
                for (size_t j = 0; j < GRID_COLUMNS; j++)
                {
                    float frequency = 1.0f;
                    float amplitude = 1.0f;
                    float noise = 0.0f;

                    for (size_t k = 0; k < octaveCount; k++)
                    {
                        noise += sample(geometry::Point2(
                                     j * frequency / BASE_FREQUENCY,
                                     i * frequency / BASE_FREQUENCY)) *
                                 amplitude;
                        frequency *= 2.0f;
                        amplitude *= 0.5f;
                    }
                    noise *= 1.2f;    // Add contrast.
                    if (noise > 1.0f) // Clip noise.
                    {
                        noise = 1.0f;
                    }
                    else if (noise < -1.0f)
                    {
                        noise = -1.0f;
                    }
                    uint8_t color = (uint8_t)((noise + 1.0f) * 127.5f);
                    grid[i][j] = color;
                }
            }
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        int grid[GRID_ROWS][GRID_COLUMNS];
        uint seed;

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        __host__ __device__ float cubicInterpolation(float a0, float a1, float w) const
        {
            return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
        }

        __host__ __device__ geometry::Vec2 randomGradient(geometry::Point2 point) const
        {
            const unsigned w = 8 * sizeof(unsigned);
            const unsigned s = w / 2;
            unsigned a = point.x();
            unsigned b = point.y();

            a ^= seed;
            b ^= seed;

            b ^= a << s | a >> w - s;
            b *= 1911520717;

            a ^= b << s | b >> w - s;
            a *= 2048419325;
            float random = a * (3.14159265 / ~(~0u >> 1)); // in [0, 2*Pi].

            return geometry::Vec2(sinf(random), cosf(random));
        }

        __host__ __device__ float dotGridGradient(geometry::Point2 corner, geometry::Point2 point) const
        {
            geometry::Vec2 gradient = randomGradient(corner);
            geometry::Vec2 distance = point - corner;
            return dot(gradient, distance);
        }

        __host__ __device__ float sample(geometry::Point2 point) const
        {
            /*
            Given a grid with dimensions rows x columns, where each cell is formed
            at the intersection of a row and a column, and each edge of a row and
            column serves as a point. The coordinates x,y are defined relative to the grid.
            *
            *   0__1__2__3__...
            * 0 |__|__|__|__...
            * 1 |__|__|__|__...
            * 2 |__|__|__|__...
            *  ⋮ ⋮ ⋮  ⋮ ⋮
            *
            */

            // Determine grid cell coordinates.
            int x0 = (int)point.x();
            int y0 = (int)point.y();
            // Get the other side of the cell.
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float interpolationWeightX = point.x() - (float)x0;
            float interpolationWeightY = point.y() - (float)y0;

            // Compute and interpolate top two corners.
            float n0 = dotGridGradient(geometry::Point2(x0, y0), point);
            float n1 = dotGridGradient(geometry::Point2(x1, y0), point);
            float ix0 = cubicInterpolation(n0, n1, interpolationWeightX);

            // Compute and interpolate bottom two corners.
            n0 = dotGridGradient(geometry::Point2(x0, y1), point);
            n1 = dotGridGradient(geometry::Point2(x1, y1), point);
            float ix1 = cubicInterpolation(n0, n1, interpolationWeightX);

            return cubicInterpolation(ix0, ix1, interpolationWeightY);
        }

        __host__ __device__ graphics::RGBPixel getTerrainColor(graphics::ColorChannel intensity)
        {

            const graphics::ColorChannel DEEP_WATER = 43;
            const graphics::ColorChannel SHALLOW_WATER = 86;
            const graphics::ColorChannel BEACH = 129;
            const graphics::ColorChannel LOWLANDS = 172;
            const graphics::ColorChannel HIGHLANDS = 215;

            if (intensity < DEEP_WATER)
            {
                graphics::RGBPixel darkBlue(0, 0, intensity + 212);
                return darkBlue;
            }
            else if (intensity < SHALLOW_WATER)
            {
                graphics::RGBPixel lightBlue(0, intensity, 255);
                return lightBlue;
            }
            else if (intensity < BEACH)
            {
                graphics::RGBPixel sandy(255, 255 - (intensity - 86), 128);
                return sandy;
            }
            else if (intensity < LOWLANDS)
            {
                graphics::RGBPixel lightGreen(128, intensity - 86, 64);
                return lightGreen;
            }
            else if (intensity < HIGHLANDS)
            {
                graphics::RGBPixel darkGreen(intensity - 86, 200, intensity - 128);
                return darkGreen;
            }
            else // Peaks
            {
                graphics::RGBPixel white(255, 255, 255);
                return white;
            }
        }

        // ----------------------------------------------------------------
        // --- Private attributes
    };
} // namespace procedural_generation

#endif // PERLINNOISEGENERATOR_CUH