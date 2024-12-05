#include "Timer.cuh"

__host__ Timer::Timer() : m_start(0), m_end(0) {}

__host__ void Timer::resume()
{
    m_start = clock();
}

__host__ void Timer::pause()
{
    m_end = clock();
}

__host__ void Timer::reset()
{
    m_start = 0;
    m_end = 0;
}

__host__ double Timer::elapsed(TimeUnit timeUnit)
{
    double elapsed = static_cast<double>(m_end - m_start);
    switch (timeUnit)
    {
    case TimeUnit::SECONDS:
        elapsed /= CLOCKS_PER_SEC;
        break;
    case TimeUnit::MILLISECONDS:
        elapsed /= (CLOCKS_PER_SEC / 1000);
        break;
    case TimeUnit::CLOCK_TICKS:
        break;
    default:
        break;
    }
    return elapsed;
}

__host__ double Timer::print(std::string task, TimeUnit timeUnit)
{
    double elapsed = this->elapsed(timeUnit);
    std::cout << task << " took " << elapsed << " ";
    switch (timeUnit)
    {
    case TimeUnit::SECONDS:
        std::cout << "seconds\n";
        break;
    case TimeUnit::MILLISECONDS:
        std::cout << "milliseconds\n";
        break;
    case TimeUnit::CLOCK_TICKS:
        std::cout << "clock ticks\n";
        break;
    default:
        break;
    }
    return elapsed;
}