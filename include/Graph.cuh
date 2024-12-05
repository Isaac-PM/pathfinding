#ifndef GRAPH_CUH
#define GRAPH_CUH

#include "PerlinNoiseGenerator.cuh"
#include <array>
#include <sstream>
#include <utility>
#include <vector>

namespace graph
{
    using VertexID = uint;
    using Weight = uint;

    class Graph // Undirected graph.
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static constexpr Weight INFINITE_WEIGHT = std::numeric_limits<Weight>::max();
        static constexpr VertexID INVALID_VERTEX = std::numeric_limits<VertexID>::max();
        static constexpr std::array<std::pair<int, int>, 8> PATHFINDING_GRID_DIRECTIONS =
            {{
                {-1, 0},  // Up.
                {1, 0},   // Down.
                {0, -1},  // Left.
                {0, 1},   // Right.
                {-1, -1}, // Up-left diagonal.
                {-1, 1},  // Up-right diagonal.
                {1, -1},  // Down-left diagonal.
                {1, 1}    // Down-right diagonal.
            }};

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ Graph() {}

        [[nodiscard]] __host__ static Graph *fromPerlinNoise(
            const procedural_generation::PerlinNoiseGenerator &generator);

        __host__ Weight getWeight(VertexID vertex, VertexID neighbor) const;

        __host__ std::string toString() const;

        /*
        Converts the graph into a pathfinding grid representation.
        Requires the graph to originate from a grid structure,
        meaning each node must have exactly 8 ordered neighbors.
        This method is not compatible with arbitrary graphs and
        is closely tied to the Perlin Noise Generator class.
        Following functions are related to this context:
        */
        using PathfindingGrid = std::vector<std::vector<Weight>>;

        __host__ PathfindingGrid asGrid(size_t rows, size_t columns) const;

        __host__ __device__ static bool isValidDirection(int row, int column, size_t rows, size_t columns)
        {
            // Check if the direction is within the bounds of the grid.
            return row >= 0 && row < rows && column >= 0 && column < columns;
        }

        __host__ static std::pair<size_t, size_t> indexToCoordinates(size_t index, size_t columns)
        {
            // Transform a 1D index to 2D coordinates.
            size_t row = index / columns;
            size_t column = index % columns;
            return std::make_pair(row, column);
        }

        __host__ static size_t coordinatesToIndex(size_t row, size_t column, size_t columns)
        {
            // Transform 2D coordinates to a 1D index.
            return row * columns + column;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        std::vector<VertexID> adjacencyList; // Neighbors of each vertex.
        /*
         *
         *  45, 89, 23, ..., 21, 34, ..., 66, ...
         *  ^                ^            ^
         *  |0_______________|1___________|2_____ Neighbors for the vertices 0, 1, 2, ...
         *
         */
        std::vector<Weight> weights;    // Weights of each edge.
        std::vector<uint> edgesOffsets; // Offset to the adjacency list of each vertex.
        std::vector<uint> edgesSize;    // Number of edges of each vertex.
        size_t vertexCount;
        size_t edgeCount;

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods

        // ----------------------------------------------------------------
        // --- Private attributes
    };
} // namespace graph

#endif // GRAPH_CUH