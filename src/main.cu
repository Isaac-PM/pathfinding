#include "Graph.cuh"
#include "Pathfinding.cuh"
#include "PerlinNoiseGenerator.cuh"
#include "PPMImage.cuh"
#include "Timer.cuh"

using namespace benchmarking;
using namespace geometry;
using namespace graph;
using namespace graphics;
using namespace procedural_generation;

enum PathGenerationMethod
{
    CPU_DIJKSTRA,
    CUDA_DIJKSTRA,
    CPU_A_STAR,
    CUDA_A_STAR
};

__host__ inline void paintPath(const std::vector<VertexID> &path, PPMImage *image, PathGenerationMethod method)
{
    const RGBPixel CPU_PATH_COLOR = {0, 255, 255};  // Cyan.
    const RGBPixel CUDA_PATH_COLOR = {255, 0, 255}; // Magenta.

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
                    if (method == CPU_DIJKSTRA || method == CPU_A_STAR)
                    {
                        image->setPixel(x, y, CPU_PATH_COLOR);
                    }
                    else
                    {
                        image->setPixel(x, y, CUDA_PATH_COLOR);
                    }
                }
            }
        }
    }
}

int main(void)
{
    // nodos aleatorios

    PerlinNoiseGenerator *map = PerlinNoiseGenerator::generateAndSave(); // Represents a 2D topographical map.
    Graph *graph = Graph::fromPerlinNoise(*map);                         // Generate a graph based on the Perlin noise map.

    VertexID startingVertex = 0;                                                                      // Top-left corner of the map.
    VertexID endingVertex = PerlinNoiseGenerator::GRID_ROWS * PerlinNoiseGenerator::GRID_COLUMNS - 1; // Bottom-right corner of the map.

    Timer timer;
    std::vector<VertexID> cudaShortestPath = pathfinding::dijkstraCUDA(*graph, startingVertex, endingVertex, timer);
    std::cout << "CUDA Dijkstra's algorithm took " << timer.elapsed(TimeUnit::MILLISECONDS) << " ms\n";
    std::vector<VertexID> cpuShortestPath = pathfinding::flowFieldPathfindingAccelerated(*graph, *map, {0, 0}, {999, 999}, timer);
    std::cout << "CPU Dijkstra's algorithm took " << timer.elapsed(TimeUnit::MILLISECONDS) << " ms\n";

    PPMImage *graphAsImage = map->asTerrainLikeImage();
    paintPath(cudaShortestPath, graphAsImage, CUDA_DIJKSTRA);
    graphAsImage->save("path_dijkstra_cuda.ppm");
    delete graphAsImage;

    graphAsImage = map->asTerrainLikeImage();
    paintPath(cpuShortestPath, graphAsImage, CPU_DIJKSTRA);
    graphAsImage->save("path_dijkstra_cpu.ppm");
    delete graphAsImage;

    delete map;
    delete graph;
    return EXIT_SUCCESS;
}
