#include "PPMImage.cuh"

namespace graphics
{
    __host__ __device__ RGBPixel::RGBPixel(ColorChannel r, ColorChannel g, ColorChannel b)
        : r(r), g(g), b(b) {}

    __host__ PPMImage::PPMImage(
        size_t columns,
        float aspectRatio)
        : m_columns(columns),
          m_aspectRatio(aspectRatio),
          m_maxChannelValue(RGBPixel::MAX_VALUE)
    {
        m_rows = size_t(m_columns / m_aspectRatio);
        m_rows = (m_rows < 1 ? 1 : m_rows);
        m_pixelCount = m_rows * m_columns;
        m_pixels = new RGBPixel[m_pixelCount];
        for (size_t i = 0; i < m_pixelCount; ++i)
        {
            m_pixels[i] = RGBPixel();
        }
    }

    __host__ PPMImage::PPMImage(
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
        for (size_t i = 0; i < m_pixelCount; ++i)
        {
            m_pixels[i] = RGBPixel();
        }
    }

    __host__ PPMImage::~PPMImage()
    {
        delete[] m_pixels;
    }

    __host__ __device__ size_t PPMImage::rows() const
    {
        return m_rows;
    }

    __host__ __device__ size_t PPMImage::height() const
    {
        return m_rows;
    }

    __host__ __device__ size_t PPMImage::columns() const
    {
        return m_columns;
    }

    __host__ __device__ size_t PPMImage::width() const
    {
        return m_columns;
    }

    __host__ __device__ size_t PPMImage::pixelCount() const
    {
        return m_pixelCount;
    }

    __host__ __device__ RGBPixel PPMImage::getPixel(size_t row, size_t column) const
    {
        return m_pixels[row * m_columns + column];
    }

    __host__ __device__ void PPMImage::setPixel(size_t row, size_t column, const RGBPixel &pixel)
    {
        m_pixels[row * m_columns + column] = pixel;
    }

    __host__ PPMImage PPMImage::load(const std::string &path)
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

        for (size_t i = 0; i < image.m_pixelCount; ++i)
        {
            int r, g, b;
            file >> r >> g >> b;
            image.m_pixels[i] = RGBPixel((ColorChannel)r, (ColorChannel)g, (ColorChannel)b);
        }

        file.close();
        return image;
    }

    __host__ void PPMImage::save(const std::string &path) const
    {
        std::fstream file(path, std::ios::out);
        if (!file.is_open())
        {
            throw std::runtime_error("File not found");
        }

        file << FILE_FORMAT_PPM << '\n';
        file << m_columns << ' ' << m_rows << '\n';
        file << m_maxChannelValue << '\n';

        for (size_t i = 0; i < m_pixelCount; ++i)
        {
            file << m_pixels[i] << '\n';
        }

        file.close();
    }
} // namespace graphics