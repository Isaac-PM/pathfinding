#include "PerlinNoiseGenerator.cuh"

namespace procedural_generation
{
    __host__ PerlinNoiseGenerator::PerlinNoiseGenerator(uint userSeed)
    {
        seed = userSeed != 0 ? userSeed : (uint)time(nullptr);
    }

    [[nodiscard]] __host__ PerlinNoiseGenerator *PerlinNoiseGenerator::generateAndSave(
        uint octaveCount,
        const std::string &outputName)
    {
        PerlinNoiseGenerator *generator = new PerlinNoiseGenerator();
        generator->generate(octaveCount);
        generator->save(outputName);
        return generator;
    }

    __host__ graphics::PPMImage *PerlinNoiseGenerator::asBlackAndWhiteImage() const
    {
        using namespace graphics;
        PPMImage *image = new PPMImage(GRID_ROWS, GRID_COLUMNS, 1.0f);
        for (size_t i = 0; i < GRID_ROWS; ++i)
        {
            for (size_t j = 0; j < GRID_COLUMNS; ++j)
            {
                ColorChannel intensity = grid[i][j];
                RGBPixel pixel(intensity, intensity, intensity);
                image->setPixel(i, j, pixel);
            }
        }
        return image;
    }

    __host__ graphics::PPMImage *PerlinNoiseGenerator::asTerrainLikeImage() const
    {
        using namespace graphics;
        PPMImage *image = new PPMImage(GRID_ROWS, GRID_COLUMNS, 1.0f);
        for (size_t i = 0; i < GRID_ROWS; ++i)
        {
            for (size_t j = 0; j < GRID_COLUMNS; ++j)
            {
                ColorChannel intensity = grid[i][j];
                RGBPixel pixel = getTerrainColor(intensity);
                image->setPixel(i, j, pixel);
            }
        }
        return image;
    }

    __host__ void PerlinNoiseGenerator::save(const std::string &outputName)
    {
        using namespace graphics;
        std::string blackAndWhitePath = outputName + ".ppm";
        std::string terrainLikePath = outputName + "_terrain.ppm";

        PPMImage *blackAndWhiteImage = asBlackAndWhiteImage();
        PPMImage *terrainLikeImage = asTerrainLikeImage();

        blackAndWhiteImage->save(blackAndWhitePath);
        terrainLikeImage->save(terrainLikePath);

        delete blackAndWhiteImage;
        delete terrainLikeImage;
    }

    __host__ __device__ void PerlinNoiseGenerator::generate(uint octaveCount)
    {
        constexpr uint BASE_FREQUENCY = (GRID_ROWS * GRID_COLUMNS) / 5184; // Based on the proportion used in the video.
        for (size_t i = 0; i < GRID_ROWS; ++i)
        {
            for (size_t j = 0; j < GRID_COLUMNS; ++j)
            {
                float frequency = 1.0f;
                float amplitude = 1.0f;
                float noise = 0.0f;

                for (size_t k = 0; k < octaveCount; ++k)
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

    __host__ __device__ float PerlinNoiseGenerator::cubicInterpolation(float a0, float a1, float w) const
    {
        return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
    }

    __host__ __device__ geometry::Vec2 PerlinNoiseGenerator::randomGradient(geometry::Point2 point) const
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

    __host__ __device__ float PerlinNoiseGenerator::dotGridGradient(geometry::Point2 corner, geometry::Point2 point) const
    {
        geometry::Vec2 gradient = randomGradient(corner);
        geometry::Vec2 distance = point - corner;
        return dot(gradient, distance);
    }

    __host__ __device__ float PerlinNoiseGenerator::sample(geometry::Point2 point) const
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

    __host__ graphics::RGBPixel PerlinNoiseGenerator::getTerrainColor(graphics::ColorChannel intensity)
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
} // namespace procedural_generation