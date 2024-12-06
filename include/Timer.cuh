#ifndef TIMER_H
#define TIMER_H

#include <iostream>

namespace benchmarking
{
    enum TimeUnit
    {
        SECONDS,
        MILLISECONDS,
        CLOCK_TICKS
    };

    class Timer
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ Timer();

        __host__ ~Timer() = default;

        __host__ void resume();

        __host__ void pause();

        __host__ void reset();

        __host__ double elapsed(TimeUnit timeUnit = TimeUnit::SECONDS);

        __host__ double print(std::string task, TimeUnit timeUnit = TimeUnit::MILLISECONDS);

        // ----------------------------------------------------------------
        // --- Public attributes

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods
        clock_t m_start;
        clock_t m_end;

        // ----------------------------------------------------------------
        // --- Private attributes
    };
} // namespace benchmarking

#endif // TIMER_H