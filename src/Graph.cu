#include "Graph.cuh"

namespace graph
{
    [[nodiscard]] __host__ Graph *Graph::fromPerlinNoise(const procedural_generation::PerlinNoiseGenerator &generator)
    {
        /*
        Generates a graph based on Perlin noise, where each cell is a node and the number
        contained in that cell is the weight associated with moving to that cell.
        Each cell can have a maximum of 8 edges. The graph is undirected.
        */
        using namespace procedural_generation;
        Graph *graph = new Graph();
        size_t generalVertexCount = 0;
        size_t generalEdgeCount = 0;

        for (size_t row = 0; row < PerlinNoiseGenerator::GRID_ROWS; ++row)
        {
            for (size_t column = 0; column < PerlinNoiseGenerator::GRID_COLUMNS; ++column)
            {
                size_t vertexEdgeCount = 0;
                for (const auto &[directionRow, directionColumn] : PATHFINDING_GRID_DIRECTIONS)
                {
                    int newRow = row + directionRow;
                    int newColumn = column + directionColumn;
                    if (isValidDirection(newRow, newColumn, PerlinNoiseGenerator::GRID_ROWS, PerlinNoiseGenerator::GRID_COLUMNS))
                    {
                        VertexID neighbor = newRow * PerlinNoiseGenerator::GRID_COLUMNS + newColumn;
                        Weight weight = generator.grid[newRow][newColumn];
                        graph->adjacencyList.push_back(neighbor);
                        graph->weights.push_back(weight);
                        vertexEdgeCount++;
                    }
                }
                generalVertexCount++;
                generalEdgeCount += vertexEdgeCount;
                graph->edgesOffsets.push_back(generalEdgeCount - vertexEdgeCount);
                graph->edgesSize.push_back(vertexEdgeCount);
            }
        }
        graph->vertexCount = generalVertexCount;
        graph->edgeCount = generalEdgeCount;

        return graph;
    }

    __host__ Weight Graph::getWeight(VertexID vertex, VertexID neighbor) const
    {
        size_t offset = edgesOffsets[vertex];
        size_t size = edgesSize[vertex];
        for (size_t i = 0; i < size; ++i)
        {
            if (adjacencyList[offset + i] == neighbor)
            {
                return weights[offset + i];
            }
        }
        return INFINITE_WEIGHT;
    }

    __host__ std::string Graph::toString() const
    {
        std::ostringstream oss;
        oss << "Graph information:\n";
        oss << "Vertex count: " << vertexCount << "\n";
        oss << "Edge count: " << edgeCount << "\n";

        size_t adjacencyListSize = adjacencyList.capacity() * sizeof(uint);
        size_t weightsSize = weights.capacity() * sizeof(uint);
        size_t edgesOffsetsSize = edgesOffsets.capacity() * sizeof(uint);
        size_t edgesSizeSize = edgesSize.capacity() * sizeof(uint);
        float totalSizeMB = (adjacencyListSize + weightsSize + edgesOffsetsSize + edgesSizeSize) / 1024.0f / 1024.0f;
        oss << "Graph size (MB): " << totalSizeMB << "\n";

        float verticesToPrint = vertexCount * (0.0625f / 100.0f);
        oss << "Adjacency list (first " << verticesToPrint << " vertices):\n";
        for (size_t vertex = 0; vertex < verticesToPrint; ++vertex)
        {
            oss << "Vertex " << vertex << " -> ";
            size_t offset = edgesOffsets[vertex];
            size_t size = edgesSize[vertex];
            for (size_t i = 0; i < size; ++i)
            {
                VertexID neighborID = adjacencyList[offset + i];
                Weight weight = weights[offset + i];
                oss << "(" << neighborID << ", " << weight << ")";
                i < size - 1 ? oss << ", " : oss << "\n";
            }
        }
        return oss.str();
    }
} // namespace graph