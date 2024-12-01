#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

#ifdef _WIN32
#include <windows.h>
#define sleep(seconds) Sleep((seconds) * 1000) // Takes milliseconds.
#else
#include <unistd.h>
#define sleep(seconds) usleep((seconds) * 1000000) // Takes microseconds.
#endif

#endif // CONSTANTS_CUH