#ifndef GAME_CUH
#define GAME_CUH

#include "Pathfinding.cuh"
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <unordered_map>
#include <vector>

namespace game
{

    using namespace benchmarking;
    using namespace geometry;
    using namespace graph;
    using namespace graphics;
    using namespace pathfinding;
    using namespace procedural_generation;

    constexpr TimeUnit TIME_UNIT = TimeUnit::MILLISECONDS;
    constexpr const char *TIME_UNIT_STRING = "milliseconds";
    constexpr const char *RESULTS_FILE = "results.csv";

    enum Strategy
    {
        NONE,
        FLOW_FIELD_ACCELERATED, // Flow field pathfinding accelerated by CUDA.
        DIJKSTRA_CPU,
        DIJKSTRA_CUDA,
    };

    void savePlayerRound(
        uint round,
        const std::string &player,
        const Strategy &strategy,
        const VertexID &startingVertex,
        const VertexID &endingVertex,
        double timePlayer,
        double totalTimePlayer,
        double pointsPlayer,
        double totalPointsPlayer,
        bool playerWonRound,
        uint playerWins,
        std::ofstream &resultsFile);
    void paintPath(const std::vector<VertexID> &path, PPMImage *image, const RGBPixel &playerColor);
    int chooseBestGPU();
    int chooseRandomGPU();

    enum StrategyClassification
    {
        BEST_STRATEGY,
        AVERAGE_STRATEGY,
        WORST_STRATEGY,
    };

    std::unordered_map<StrategyClassification, Strategy> strategies = {
        {BEST_STRATEGY, NONE},
        {AVERAGE_STRATEGY, NONE},
        {WORST_STRATEGY, NONE},
    };

    struct RunResult
    {
        double time;
        std::vector<VertexID> path;
    };

    __host__ inline RunResult runStrategy(
        Strategy strategy,
        const Graph &graph,
        const PerlinNoiseGenerator &map,
        VertexID startingVertex,
        VertexID endingVertex)
    {
        Timer timer;
        std::vector<VertexID> path;
        switch (strategy)
        {
        case Strategy::FLOW_FIELD_ACCELERATED:
        {
            Coordinates startingCoordinates = Graph::indexToCoordinates(startingVertex, map.GRID_COLUMNS);
            Coordinates endingCoordinates = Graph::indexToCoordinates(endingVertex, map.GRID_COLUMNS);
            path = flowFieldPathfindingAccelerated(graph, map, startingCoordinates, endingCoordinates, timer);
            break;
        }
        case Strategy::DIJKSTRA_CPU:
        {
            path = dijkstraCPU(graph, startingVertex, endingVertex, timer);
            break;
        }
        case Strategy::DIJKSTRA_CUDA:
        {
            path = dijkstraCUDA(graph, startingVertex, endingVertex, timer);
            break;
        }
        default:
            break;
        }
        return {timer.elapsed(TIME_UNIT), path};
    }

    const uint RUNS_BY_STRATEGY = 3;

    std::unordered_map<Strategy, uint> strategyTimes = {
        {Strategy::FLOW_FIELD_ACCELERATED, 0},
        {Strategy::DIJKSTRA_CPU, 0},
        {Strategy::DIJKSTRA_CUDA, 0},
    };

    __host__ inline void benchmarkStrategies()
    {
        /*
        Tests the performance of the different pathfinding strategies,
        and classifies them as the best, average, or worst according to
        their average time.
        */
        std::cout << "Benchmarking strategies...\n";
        int bestGPU = chooseBestGPU();
        cudaSetDevice(bestGPU);
        PerlinNoiseGenerator map = PerlinNoiseGenerator();
        VertexID startingVertex = 0;                                                                      // Top-left corner of the map.
        VertexID endingVertex = PerlinNoiseGenerator::GRID_ROWS * PerlinNoiseGenerator::GRID_COLUMNS - 1; // Bottom-right corner of the map.
        for (size_t i = 0; i < RUNS_BY_STRATEGY; ++i)
        {
            std::cout << "Run " << i + 1 << "...\n";
            map.generate(PerlinNoiseGenerator::DEFAULT_OCTAVE_COUNT);
            Graph *graph = Graph::fromPerlinNoise(map);
            uint time = 0;
            for (auto &strategy : strategyTimes)
            {
                RunResult result = runStrategy(strategy.first, *graph, map, startingVertex, endingVertex);
                time += result.time;
                strategyTimes[strategy.first] += result.time;
            }
            delete graph;
        }
        for (auto &strategy : strategyTimes)
        {
            strategyTimes[strategy.first] /= RUNS_BY_STRATEGY;
        }

        std::vector<std::pair<Strategy, uint>> sortedStrategies(strategyTimes.begin(), strategyTimes.end());
        std::sort(sortedStrategies.begin(), sortedStrategies.end(), [](const auto &a, const auto &b)
                  { return a.second < b.second; });

        strategies[BEST_STRATEGY] = sortedStrategies.front().first;
        strategies[AVERAGE_STRATEGY] = sortedStrategies[sortedStrategies.size() / 2].first;
        strategies[WORST_STRATEGY] = sortedStrategies.back().first;

        std::cout << "Best strategy: " << strategies[BEST_STRATEGY] << " with an average time of " << strategyTimes[strategies[BEST_STRATEGY]] << " " << TIME_UNIT_STRING << ".\n";
        std::cout << "Average strategy: " << strategies[AVERAGE_STRATEGY] << " with an average time of " << strategyTimes[strategies[AVERAGE_STRATEGY]] << " " << TIME_UNIT_STRING << ".\n";
        std::cout << "Worst strategy: " << strategies[WORST_STRATEGY] << " with an average time of " << strategyTimes[strategies[WORST_STRATEGY]] << " " << TIME_UNIT_STRING << ".\n";
        std::cout << '\n';
    }

    const uint ROUNDS = 10;

    __host__ inline void play()
    {
        benchmarkStrategies();

        std::map<Strategy, int> pointsByStrategyPlayer1 = {
            {strategies[BEST_STRATEGY], 1}, // Initialize in 1 to avoid division by zero.
            {strategies[AVERAGE_STRATEGY], 1},
            {strategies[WORST_STRATEGY], 1},
        };
        std::map<Strategy, int> pointsByStrategyPlayer2 = {
            {strategies[BEST_STRATEGY], 1},
            {strategies[AVERAGE_STRATEGY], 1},
            {strategies[WORST_STRATEGY], 1},
        };
        uint player1Wins = 0;
        uint totalTimePlayer1 = 0;
        double totalPointsPlayer1 = 1;
        bool player1WonRound = false;
        const RGBPixel PLAYER_1_PATH_COLOR = {0, 255, 255}; // Cyan

        std::map<Strategy, uint> strategyCountPlayer1 = {
            {strategies[BEST_STRATEGY], 1},
            {strategies[AVERAGE_STRATEGY], 1},
            {strategies[WORST_STRATEGY], 1},
        };
        std::map<Strategy, uint> strategyCountPlayer2 = {
            {strategies[BEST_STRATEGY], 1},
            {strategies[AVERAGE_STRATEGY], 1},
            {strategies[WORST_STRATEGY], 1},
        };
        uint player2Wins = 0;
        uint totalTimePlayer2 = 0;
        double totalPointsPlayer2 = 1;
        bool player2WonRound = false;
        const RGBPixel PLAYER_2_PATH_COLOR = {255, 0, 255}; // Magenta

        auto getPoints = [](const RunResult &result, Strategy strategy) -> double
        {
            if (strategy == strategies[BEST_STRATEGY])
            {
                return 10; // Penalize the usage of the best strategy.
            }
            else if (strategy == strategies[AVERAGE_STRATEGY])
            {
                return 20;
            }
            else if (strategy == strategies[WORST_STRATEGY])
            {
                return 30 - (result.time / 100); // Penalize the worst strategy according to its time.
            }
            return 0;
        };

        auto chooseStrategy = [](const std::map<Strategy, int> &pointsByStrategyPlayer, const std::map<Strategy, uint> &strategyCount, double epsilon = 0.1) -> Strategy
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> distribution(0.0, 1.0);
            if (distribution(gen) < epsilon)
            { // Exploration.
                std::uniform_int_distribution<> choice(0, strategies.size() - 1);
                return strategies[static_cast<StrategyClassification>(choice(gen))];
            }
            else
            { // Exploitation.
                Strategy optimalStrategy = strategies[BEST_STRATEGY];
                double bestPoints = pointsByStrategyPlayer.at(optimalStrategy) / strategyCount.at(optimalStrategy);
                for (const auto &strategy : strategies)
                {
                    double points = pointsByStrategyPlayer.at(strategy.second) / strategyCount.at(strategy.second);
                    if (points > bestPoints)
                    {
                        bestPoints = points;
                        optimalStrategy = strategy.second;
                    }
                }
                return optimalStrategy;
            }
        };

        std::random_device rd;
        std::mt19937 gen(rd());
        int gpu = -1;
        PPMImage *graphAsImage = nullptr;
        std::ofstream resultsFile(RESULTS_FILE);
        resultsFile << "Round," << "Player,"
                    << "Starting Row," << "Starting Column,"
                    << "Ending Row," << "Ending Column,"
                    << "Strategy,"
                    << "Time (" << TIME_UNIT_STRING << ")," << "Total Time (" << TIME_UNIT_STRING << "),"
                    << "Points," << "Total Points,"
                    << "Player Won Round," << "Player Wins\n";

        for (size_t round = 1; round <= ROUNDS; ++round)
        {
            std::cout << "Round " << round << "...\n";
            PerlinNoiseGenerator map = PerlinNoiseGenerator();
            map.generate(PerlinNoiseGenerator::DEFAULT_OCTAVE_COUNT);
            Graph *graph = Graph::fromPerlinNoise(map);
            VertexID startingVertex = 0;
            VertexID endingVertex = 0;
            while (startingVertex == endingVertex)
            {
                startingVertex = std::uniform_int_distribution<VertexID>(0, map.GRID_ROWS * map.GRID_COLUMNS - 1)(gen);
                endingVertex = std::uniform_int_distribution<VertexID>(0, map.GRID_ROWS * map.GRID_COLUMNS - 1)(gen);
            }
            player1WonRound = false;
            player2WonRound = false;

            gpu = chooseRandomGPU();
            cudaSetDevice(gpu);
            Strategy strategyPlayer1 = chooseStrategy(pointsByStrategyPlayer1, strategyCountPlayer1);
            std::cout << "Player 1: " << strategyPlayer1 << "...\n";
            RunResult resultPlayer1 = runStrategy(strategyPlayer1, *graph, map, startingVertex, endingVertex);
            double pointsPlayer1 = getPoints(resultPlayer1, strategyPlayer1);
            totalTimePlayer1 += resultPlayer1.time;
            pointsByStrategyPlayer1[strategyPlayer1] += pointsPlayer1;
            totalPointsPlayer1 += pointsPlayer1;
            strategyCountPlayer1[strategyPlayer1] += 1;
            graphAsImage = map.asTerrainLikeImage();
            paintPath(resultPlayer1.path, graphAsImage, PLAYER_1_PATH_COLOR);
            graphAsImage->save("path_player1_round_" + std::to_string(round) + ".ppm");
            std::cout << "Player 1: " << strategyPlayer1 << " with " << pointsPlayer1 << " points.\n";
            delete graphAsImage;

            gpu = chooseRandomGPU();
            cudaSetDevice(gpu);
            Strategy strategyPlayer2 = chooseStrategy(pointsByStrategyPlayer2, strategyCountPlayer2);
            std::cout << "Player 2: " << strategyPlayer2 << "...\n";
            RunResult resultPlayer2 = runStrategy(strategyPlayer2, *graph, map, startingVertex, endingVertex);
            double pointsPlayer2 = getPoints(resultPlayer2, strategyPlayer2);
            totalTimePlayer2 += resultPlayer2.time;
            pointsByStrategyPlayer2[strategyPlayer2] += pointsPlayer2;
            totalPointsPlayer2 += pointsPlayer2;
            strategyCountPlayer2[strategyPlayer2] += 1;
            graphAsImage = map.asTerrainLikeImage();
            paintPath(resultPlayer2.path, graphAsImage, PLAYER_2_PATH_COLOR);
            graphAsImage->save("path_player2_round_" + std::to_string(round) + ".ppm");
            std::cout << "Player 2: " << strategyPlayer2 << " with " << pointsPlayer2 << " points.\n";
            delete graphAsImage;

            if (pointsPlayer1 > pointsPlayer2)
            {
                player1Wins += 1;
                player1WonRound = true;
            }
            else if (pointsPlayer2 > pointsPlayer1)
            {
                player2Wins += 1;
                player2WonRound = true;
            }
            savePlayerRound(
                round,
                "Player 1",
                strategyPlayer1,
                startingVertex,
                endingVertex,
                resultPlayer1.time,
                totalTimePlayer1,
                pointsPlayer1,
                totalPointsPlayer1,
                player1WonRound,
                player1Wins,
                resultsFile);

            savePlayerRound(
                round,
                "Player 2",
                strategyPlayer2,
                startingVertex,
                endingVertex,
                resultPlayer2.time,
                totalTimePlayer2,
                pointsPlayer2,
                totalPointsPlayer2,
                player2WonRound,
                player2Wins,
                resultsFile);

            delete graph;
            std::cout << '\n';
        }
        resultsFile.close();
    }

    __host__ inline void savePlayerRound(
        uint round,
        const std::string &player,
        const Strategy &strategy,
        const VertexID &startingVertex,
        const VertexID &endingVertex,
        double timePlayer,
        double totalTimePlayer,
        double pointsPlayer,
        double totalPointsPlayer,
        bool playerWonRound,
        uint playerWins,
        std::ofstream &resultsFile)
    {
        Coordinates startingCoordinates = Graph::indexToCoordinates(startingVertex, PerlinNoiseGenerator::GRID_COLUMNS);
        Coordinates endingCoordinates = Graph::indexToCoordinates(endingVertex, PerlinNoiseGenerator::GRID_COLUMNS);
        resultsFile
            << round << ","
            << player << ","
            << startingCoordinates.row << "," << startingCoordinates.column << ","
            << endingCoordinates.row << "," << endingCoordinates.column << ","
            << strategy << ","
            << timePlayer << ","
            << totalTimePlayer << ","
            << pointsPlayer << ","
            << totalPointsPlayer << ","
            << playerWonRound << ","
            << playerWins << "\n";
    }

    __host__ inline void paintPath(const std::vector<VertexID> &path, PPMImage *image, const RGBPixel &playerColor)
    {
        for (VertexID vertex : path)
        {
            Coordinates coordinates = Graph::indexToCoordinates(vertex, image->columns());
            for (int dx = -1; dx <= 1; ++dx) // Draw a 3x3 square around the vertex.
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    size_t x = coordinates.row + dx;
                    size_t y = coordinates.column + dy;
                    if (x < image->columns() && y < image->rows())
                    {
                        image->setPixel(x, y, playerColor);
                    }
                }
            }
        }
    }

    __host__ inline int chooseBestGPU()
    {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found!\n";
            return -1;
        }
        int bestDevice = -1;
        int maxComputeCapability = 0;
        size_t maxMemory = 0;
        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            int computeCapability = deviceProp.major * 10 + deviceProp.minor;
            size_t totalMemory = deviceProp.totalGlobalMem;
            if (computeCapability > maxComputeCapability ||
                (computeCapability == maxComputeCapability && totalMemory > maxMemory))
            {
                bestDevice = i;
                maxComputeCapability = computeCapability;
                maxMemory = totalMemory;
            }
        }
        return bestDevice;
    }

    __host__ inline int chooseRandomGPU()
    {
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0)
        {
            std::cerr << "No CUDA devices found!\n";
            return -1;
        }
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        int randomDevice = std::rand() % deviceCount;
        return randomDevice;
    }
} // namespace game

#endif // GAME_CUH