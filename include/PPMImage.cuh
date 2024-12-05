#ifndef PPMIMAGE_H
#define PPMIMAGE_H

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <string>

namespace graphics
{
    using ColorChannel = uint8_t;

    class RGBPixel
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static const size_t MAX_VALUE = 255;
        static const size_t MIN_VALUE = 0;

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ __device__ RGBPixel(ColorChannel r = 0, ColorChannel g = 0, ColorChannel b = 0);

        __host__ __device__ bool operator==(const RGBPixel &other) const
        {
            return r == other.r && g == other.g && b == other.b;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        ColorChannel r;
        ColorChannel g;
        ColorChannel b;

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const RGBPixel &p)
    {
        return out << static_cast<int>(p.r) << ' ' << static_cast<int>(p.g) << ' ' << static_cast<int>(p.b);
    }

    class PPMImage
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static const size_t DEFAULT_COLUMNS = 400;
        static constexpr float DEFAULT_ASPECT_RATIO = 16.0f / 9.0f;
        static constexpr const char *FILE_FORMAT_PPM = "P3";

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ PPMImage(
            size_t columns = DEFAULT_COLUMNS,
            float aspectRatio = DEFAULT_ASPECT_RATIO);

        __host__ PPMImage(
            size_t rows,
            size_t columns,
            float aspectRatio);

        __host__ ~PPMImage();

        __host__ __device__ size_t rows() const;

        __host__ __device__ size_t height() const;

        __host__ __device__ size_t columns() const;

        __host__ __device__ size_t width() const;

        __host__ __device__ size_t pixelCount() const;

        __host__ __device__ RGBPixel getPixel(size_t row, size_t column) const;

        __host__ __device__ void setPixel(size_t row, size_t column, const RGBPixel &pixel);

        __host__ static PPMImage load(const std::string &path);

        __host__ void save(const std::string &path) const;

        // ----------------------------------------------------------------
        // --- Public attributes

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
        size_t m_rows;
        size_t m_columns;
        size_t m_pixelCount;
        float m_aspectRatio;
        RGBPixel *m_pixels;
        size_t m_maxChannelValue;
    };
} // namespace graphics

#endif // PPMIMAGE_H