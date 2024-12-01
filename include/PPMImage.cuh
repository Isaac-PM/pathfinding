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
        __host__ __device__ RGBPixel(ColorChannel r = 0, ColorChannel g = 0, ColorChannel b = 0)
            : r(r), g(g), b(b) {}

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
            float aspectRatio = DEFAULT_ASPECT_RATIO)
            : m_columns(columns),
              m_aspectRatio(aspectRatio),
              m_maxChannelValue(RGBPixel::MAX_VALUE)
        {
            m_rows = size_t(m_columns / m_aspectRatio);
            m_rows = (m_rows < 1 ? 1 : m_rows);
            m_pixelCount = m_rows * m_columns;
            m_pixels = new RGBPixel[m_pixelCount];
            for (size_t i = 0; i < m_pixelCount; i++)
            {
                m_pixels[i] = RGBPixel();
            }
        }

        __host__ PPMImage(
            size_t rows,
            size_t columns,
            float aspectRatio)
            : m_rows(rows),
              m_columns(columns),
              m_aspectRatio(aspectRatio),
              m_maxChannelValue(RGBPixel::MAX_VALUE)
        {
            m_pixelCount = m_rows * m_columns;
            m_pixels = new RGBPixel[m_pixelCount];
            for (size_t i = 0; i < m_pixelCount; i++)
            {
                m_pixels[i] = RGBPixel();
            }
        }

        __host__ ~PPMImage()
        {
            delete[] m_pixels;
        }

        __host__ __device__ size_t rows() const
        {
            return m_rows;
        }

        __host__ __device__ size_t height() const
        {
            return m_rows;
        }

        __host__ __device__ size_t columns() const
        {
            return m_columns;
        }

        __host__ __device__ size_t width() const
        {
            return m_columns;
        }

        __host__ __device__ size_t pixelCount() const
        {
            return m_pixelCount;
        }

        __host__ __device__ RGBPixel getPixel(size_t row, size_t column) const
        {
            return m_pixels[row * m_columns + column];
        }

        __host__ __device__ void setPixel(size_t row, size_t column, const RGBPixel &pixel)
        {
            m_pixels[row * m_columns + column] = pixel;
        }

        __host__ static PPMImage load(const std::string &path)
        {

            std::fstream file(path, std::ios::in);
            if (!file.is_open())
            {
                throw std::runtime_error("File not found");
            }

            std::string imageFormat;
            getline(file, imageFormat, '\n');
            if (imageFormat != FILE_FORMAT_PPM)
            {
                throw std::runtime_error("Invalid PPM format");
            }

            PPMImage image;
            file >> image.m_columns >> image.m_rows >> image.m_maxChannelValue;
            if (image.m_maxChannelValue != RGBPixel::MAX_VALUE)
            {
                throw std::runtime_error("Invalid channel value");
            }

            for (size_t i = 0; i < image.m_pixelCount; i++)
            {
                int r, g, b;
                file >> r >> g >> b;
                image.m_pixels[i] = RGBPixel((ColorChannel)r, (ColorChannel)g, (ColorChannel)b);
            }

            file.close();
            return image;
        }

        __host__ void save(const std::string &path) const
        {
            std::fstream file(path, std::ios::out);
            if (!file.is_open())
            {
                throw std::runtime_error("File not found");
            }

            file << FILE_FORMAT_PPM << '\n';
            file << m_columns << ' ' << m_rows << '\n';
            file << m_maxChannelValue << '\n';

            for (size_t i = 0; i < m_pixelCount; i++)
            {
                file << m_pixels[i] << '\n';
            }

            file.close();
        }

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