#ifndef PATHFINDING_CUH
#define PATHFINDING_CUH

#include "Graph.cuh"
#include "PerlinNoiseGenerator.cuh"
#include "Timer.cuh"
#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <queue>
#include <vector>
#include <memory>

#define cudaCheckError()                                                                 \
    {                                                                                    \
        cudaError_t e = cudaGetLastError();                                              \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

// __host__ __device__ inline size_t toIndex(size_t row, size_t column, size_t columns)
// {
//     // Transform 2D coordinates to a 1D index.
//     return row * columns + column;
// }

namespace pathfinding
{
    using namespace graph;
    using namespace procedural_generation;
    using namespace graphics;

    const size_t MAX_THREADS_PER_BLOCK = 1024;

    // ----------------------------------------------------------------
    // --- Vector field pathfinding algorithm CPU + CUDA

    __global__ inline void xxxxxx()
    {
    }

    [[nodiscard]] __host__ inline std::vector<VertexID> vectorFieldPathfindingCUDA(
        const Graph &graph,
        size_t pathfindingGridRows,
        size_t pathfindingGridColumns,
        Timer &timer)
    {
        std::pair<size_t, size_t> startingCoordinates = Graph::indexToCoordinates(0, 0); // Top-left corner.
        std::pair<size_t, size_t> endingCoordinates =
            Graph::indexToCoordinates(pathfindingGridRows - 1, pathfindingGridColumns - 1); // Bottom-right corner.

        struct OpenListElement
        {
            size_t vertexRow;
            size_t vertexColumn;
            VertexID vertexID;
            Weight distanceToGoal;
            bool visited;
            OpenListElement(size_t row, size_t column, VertexID id, Weight distance, bool visited)
                : vertexRow(row), vertexColumn(column), vertexID(id), distanceToGoal(distance), visited(visited) {}
        };
        std::queue<std::shared_ptr<OpenListElement>> openListQueue;
        std::shared_ptr<OpenListElement> goal = std::make_shared<OpenListElement>(endingCoordinates.first, endingCoordinates.second, Graph::coordinatesToIndex(endingCoordinates.first, endingCoordinates.second, pathfindingGridColumns), 0, false);
        openListQueue.push(goal);

        std::vector<std::shared_ptr<OpenListElement>> openListVector;
        openListVector.resize(graph.vertexCount);
        for (size_t i = 0; i < graph.vertexCount; ++i)
        {
            std::pair<size_t, size_t> coordinates = Graph::indexToCoordinates(i, pathfindingGridColumns);
            size_t row = coordinates.first;
            size_t column = coordinates.second;
            if (row == endingCoordinates.first && column == endingCoordinates.second)
            {
                openListVector[i] = goal;
            }
            else
            {
                openListVector[i] = (std::make_shared<OpenListElement>(row, column, i, 0, false));
            }
        }

        Graph::PathfindingGrid pathfindingGrid = graph.asGrid(pathfindingGridRows, pathfindingGridColumns);

        while (!openListQueue.empty())
        {
            std::shared_ptr<OpenListElement> current = openListQueue.front();
            openListQueue.pop();
            size_t vertexIndex = current->vertexID;
            for (const auto &[directionRow, directionColumn] : Graph::PATHFINDING_GRID_DIRECTIONS)
            {
                int neighborRow = current->vertexRow + directionRow;
                int neighborColumn = current->vertexColumn + directionColumn;
                if (Graph::isValidDirection(neighborRow, neighborColumn, pathfindingGridRows, pathfindingGridColumns))
                {
                    size_t neighborIndex = Graph::coordinatesToIndex(neighborRow, neighborColumn, pathfindingGridColumns);
                    std::shared_ptr<OpenListElement> neighbor = openListVector[neighborIndex];

                    if (current->distanceToGoal > neighbor->distanceToGoal + 1)
                    {
                        current->distanceToGoal = neighbor->distanceToGoal + 1;
                    }
                    if (!neighbor->visited)
                    {
                        neighbor->visited = true;
                        openListQueue.push(neighbor);
                    }
                }
            }
        }

        // iterate over openListVector
        PPMImage *image = new PPMImage(pathfindingGridRows, pathfindingGridColumns, 1.0f);
        for (size_t i = 0; i < openListVector.size(); ++i)
        {
            std::shared_ptr<OpenListElement> current = openListVector[i];
            std::pair<size_t, size_t> coordinates = Graph::indexToCoordinates(i, pathfindingGridColumns);
            size_t row = coordinates.first;
            size_t column = coordinates.second;
            // scale the distance to a color
            ColorChannel intensity = current->distanceToGoal * 255 / 1000;

            RGBPixel color = {intensity, 0, 0};
            image->setPixel(column, row, color);
        }
        image->save("vector_field_pathfinding.ppm");

        return std::vector<VertexID>();
    }

    // ----------------------------------------------------------------
    // --- Vector field pathfinding algorithm in CPU

    // [[nodiscard]] __host__ inline std::vector<VertexID> vectorFieldPathfindingCPU(
    //     const Graph &graph,
    //     VertexID start,
    //     VertexID end,
    //     Timer &timer)
    // {
    // }

    // ----------------------------------------------------------------
    // --- Dijkstra's's algorithm in CUDA
    /*
    Based on https://stackoverflow.com/questions/12763991/dijkstras-algorithm-in-cuda,
    added path reconstruction and improved semantics. This version of the algorithm is
    not performant, as the weight calculations require atomic operations,
    as stated in this paper https://link.springer.com/content/pdf/10.1007/978-3-642-01970-8_91.pdf,
    the other version was not implemented.
    */

    __host__ inline void initializeArrays(
        Weight *shortestDistances,
        Weight *updatingShortestDistances,
        bool *activeVertices,
        VertexID startingVertex,
        size_t vertexCount)
    {
        shortestDistances[startingVertex] = 0;
        updatingShortestDistances[startingVertex] = 0;
        activeVertices[startingVertex] = true;

        for (size_t vertex = 1; vertex < vertexCount; ++vertex)
        {
            shortestDistances[vertex] = Graph::INFINITE_WEIGHT;
            updatingShortestDistances[vertex] = Graph::INFINITE_WEIGHT;
            activeVertices[vertex] = false;
        }
    }

    __host__ inline bool allVerticesFinalized(const bool *activeVertices, const size_t vertexCount)
    {
        for (size_t vertex = 0; vertex < vertexCount; ++vertex)
        {
            if (activeVertices[vertex])
            {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] __host__ inline std::vector<VertexID> reconstructPath(VertexID *predecessor, VertexID start, VertexID end)
    {
        std::vector<VertexID> path;
        for (VertexID vertex = end; vertex != start; vertex = predecessor[vertex])
        {
            path.push_back(vertex);
        }
        path.push_back(start);
        std::reverse(path.begin(), path.end());
        return path;
    }

    __global__ inline void relaxEdges(
        const VertexID *__restrict__ adjacencyList,
        const Weight *__restrict__ weights,
        const uint *__restrict__ edgesOffsets,
        const uint *__restrict__ edgesSize,
        const Weight *__restrict__ shortestDistances,
        Weight *__restrict__ updatingShortestDistances,
        bool *__restrict__ activeVertices,
        VertexID *__restrict__ predecessor,
        const size_t vertexCount)
    {
        size_t vertexID = blockIdx.x * blockDim.x + threadIdx.x;
        if (vertexID >= vertexCount || !activeVertices[vertexID])
        {
            return;
        }
        size_t edgesStart = edgesOffsets[vertexID];
        size_t edgesEnd = edgesStart + edgesSize[vertexID];
        for (size_t edge = edgesStart; edge < edgesEnd; ++edge)
        {
            VertexID neighbor = adjacencyList[edge];
            Weight weightToNeighbor = weights[edge];
            Weight newDistance = shortestDistances[vertexID] + weightToNeighbor;
            Weight &knownDistance = updatingShortestDistances[neighbor];
            if (atomicMin(&knownDistance, newDistance) > newDistance)
            {
                predecessor[neighbor] = vertexID;
            }
        }
        activeVertices[vertexID] = false;
    }

    __global__ inline void updateShortestDistancesAndActivateVertices(
        const VertexID *__restrict__ adjacencyList,
        const Weight *__restrict__ weights,
        const uint *__restrict__ edgesOffsets,
        const uint *__restrict__ edgesSize,
        Weight *__restrict__ shortestDistances,
        Weight *__restrict__ updatingShortestDistances,
        bool *__restrict__ activeVertices,
        const size_t vertexCount)
    {
        size_t vertexID = blockIdx.x * blockDim.x + threadIdx.x;
        if (vertexID >= vertexCount)
        {
            return;
        }
        if (shortestDistances[vertexID] > updatingShortestDistances[vertexID])
        {
            shortestDistances[vertexID] = updatingShortestDistances[vertexID];
            activeVertices[vertexID] = true;
        }
        updatingShortestDistances[vertexID] = shortestDistances[vertexID];
    }

    [[nodiscard]] __host__ inline std::vector<VertexID> dijkstraCUDA(
        const Graph &graph,
        VertexID startingVertex,
        VertexID endingVertex,
        Timer &timer)
    {
        size_t vertexCount = graph.vertexCount;

        Weight *shortestDistances;
        cudaMallocManaged(&shortestDistances, vertexCount * sizeof(Weight));
        cudaCheckError();

        Weight *updatingShortestDistances;
        cudaMallocManaged(&updatingShortestDistances, vertexCount * sizeof(Weight));
        cudaCheckError();

        /*
        Array that stores the parent (or preceding vertex)
        for each vertex in the shortest path tree. When a shorter path
        to a vertex is found during edge relaxation, the "predecessor"
        array is updated to record the vertex from which the shorter path originates.
        */
        VertexID *predecessor;
        cudaMallocManaged(&predecessor, vertexCount * sizeof(VertexID));

        /*
        A boolean array marking vertices currently active for processing.
        "true" indicates the vertex will update its neighbors in the current
        iteration, and it is reset to "false" after processing.
        */
        bool *activeVertices;
        cudaMallocManaged(&activeVertices, vertexCount * sizeof(bool));
        cudaCheckError();

        initializeArrays(shortestDistances, updatingShortestDistances, activeVertices, startingVertex, vertexCount);

        VertexID *d_adjacencyList;
        cudaMalloc(&d_adjacencyList, graph.adjacencyList.size() * sizeof(VertexID));
        cudaCheckError();
        cudaMemcpy(d_adjacencyList, graph.adjacencyList.data(), graph.adjacencyList.size() * sizeof(VertexID), cudaMemcpyHostToDevice);
        cudaCheckError();

        Weight *d_weights;
        cudaMalloc(&d_weights, graph.weights.size() * sizeof(Weight));
        cudaCheckError();
        cudaMemcpy(d_weights, graph.weights.data(), graph.weights.size() * sizeof(Weight), cudaMemcpyHostToDevice);
        cudaCheckError();

        uint *d_edgesOffsets;
        cudaMalloc(&d_edgesOffsets, graph.edgesOffsets.size() * sizeof(uint));
        cudaCheckError();
        cudaMemcpy(d_edgesOffsets, graph.edgesOffsets.data(), graph.edgesOffsets.size() * sizeof(uint), cudaMemcpyHostToDevice);
        cudaCheckError();

        uint *d_edgesSize;
        cudaMalloc(&d_edgesSize, graph.edgesSize.size() * sizeof(uint));
        cudaCheckError();
        cudaMemcpy(d_edgesSize, graph.edgesSize.data(), graph.edgesSize.size() * sizeof(uint), cudaMemcpyHostToDevice);
        cudaCheckError();

        dim3 dimGrid((vertexCount + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);

        timer.reset();
        timer.resume();
        while (!allVerticesFinalized(activeVertices, vertexCount))
        {
            relaxEdges<<<dimGrid, dimBlock>>>(
                d_adjacencyList,
                d_weights,
                d_edgesOffsets,
                d_edgesSize,
                shortestDistances,
                updatingShortestDistances,
                activeVertices,
                predecessor,
                vertexCount);
            cudaDeviceSynchronize();
            cudaCheckError();
            updateShortestDistancesAndActivateVertices<<<dimGrid, dimBlock>>>(
                d_adjacencyList,
                d_weights,
                d_edgesOffsets,
                d_edgesSize,
                shortestDistances,
                updatingShortestDistances,
                activeVertices,
                vertexCount);
            cudaDeviceSynchronize();
            cudaCheckError();
        }
        timer.pause();

        std::vector<VertexID> path = reconstructPath(predecessor, startingVertex, endingVertex);

        cudaFree(shortestDistances);
        cudaFree(updatingShortestDistances);
        cudaFree(activeVertices);
        cudaFree(d_adjacencyList);
        cudaFree(d_weights);
        cudaFree(d_edgesOffsets);
        cudaFree(d_edgesSize);

        return path;
    }

    // ----------------------------------------------------------------
    // --- Dijkstra's's algorithm in CPU

    [[nodiscard]] __host__ inline std::vector<VertexID> dijkstraCPU(
        const Graph &graph,
        VertexID start,
        VertexID end,
        Timer &timer)
    {
        size_t vertexCount = graph.vertexCount;
        std::vector<Weight> shortestDistances(vertexCount, Graph::INFINITE_WEIGHT);
        std::vector<VertexID> predecessor(vertexCount);
        std::vector<bool> visited(vertexCount, false);

        using QueueElement = std::pair<Weight, VertexID>;
        std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> priorityQueue;

        shortestDistances[start] = 0;
        priorityQueue.push({0, start});

        timer.reset();
        timer.resume();
        while (!priorityQueue.empty())
        {
            VertexID currentVertex = priorityQueue.top().second;
            priorityQueue.pop();

            if (visited[currentVertex])
            {
                continue;
            }

            visited[currentVertex] = true;

            size_t edgeOffset = graph.edgesOffsets[currentVertex];
            size_t edgeSize = graph.edgesSize[currentVertex];

            for (size_t i = 0; i < edgeSize; ++i)
            {
                VertexID neighbor = graph.adjacencyList[edgeOffset + i];
                Weight weight = graph.weights[edgeOffset + i];

                if (shortestDistances[currentVertex] + weight < shortestDistances[neighbor])
                {
                    shortestDistances[neighbor] = shortestDistances[currentVertex] + weight;
                    predecessor[neighbor] = currentVertex;
                    priorityQueue.push({shortestDistances[neighbor], neighbor});
                }
            }
        }
        timer.pause();
        std::vector<VertexID> path = reconstructPath(predecessor.data(), start, end);
        return path;
    }
}

#endif // PATHFINDING_CUH