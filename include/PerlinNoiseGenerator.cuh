#ifndef PERLINNOISEGENERATOR_CUH
#define PERLINNOISEGENERATOR_CUH

#include "PPMImage.cuh"
#include "Vec2.cuh"
#include <stdint.h>
#include <utility>

namespace procedural_generation
{
    struct Coordinates
    {
        size_t row;
        size_t column;
    };

    class PerlinNoiseGenerator
    {
        /*
        Based on "C++: Perlin Noise Tutorial" by Zipped @ https://www.youtube.com/watch?v=kCIaHqb60Cw
        */
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static const uint GRID_ROWS = 1000;
        static const uint GRID_COLUMNS = 1000;

        // ----------------------------------------------------------------
        // --- Public methods

        __host__ PerlinNoiseGenerator(uint userSeed = 0);

        [[nodiscard]] __host__ static PerlinNoiseGenerator *generateAndSave(
            uint octaveCount = 12,
            const std::string &outputName = "noise");

        __host__ graphics::PPMImage *asBlackAndWhiteImage() const;

        __host__ graphics::PPMImage *asTerrainLikeImage() const;

        __host__ void save(const std::string &outputName);

        __host__ __device__ void generate(uint octaveCount);

        // ----------------------------------------------------------------
        // --- Public attributes
        int grid[GRID_ROWS][GRID_COLUMNS];
        uint seed;

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        __host__ __device__ float cubicInterpolation(float a0, float a1, float w) const;

        __host__ __device__ geometry::Vec2 randomGradient(geometry::Point2 point) const;

        __host__ __device__ float dotGridGradient(geometry::Point2 corner, geometry::Point2 point) const;

        __host__ __device__ float sample(geometry::Point2 point) const;

        __host__ static graphics::RGBPixel getTerrainColor(graphics::ColorChannel intensity);

        // ----------------------------------------------------------------
        // --- Private attributes
    };
} // namespace procedural_generation

#endif // PERLINNOISEGENERATOR_CUH