#ifndef PATHFINDING_CUH
#define PATHFINDING_CUH

#include "Graph.cuh"
#include "PerlinNoiseGenerator.cuh"
#include "Timer.cuh"
#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

#define cudaCheckError()                                                                 \
    {                                                                                    \
        cudaError_t e = cudaGetLastError();                                              \
        if (e != cudaSuccess)                                                            \
        {                                                                                \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

namespace pathfinding
{
    using namespace benchmarking;
    using namespace geometry;
    using namespace graph;
    using namespace graphics;
    using namespace procedural_generation;

    const size_t MAX_THREADS_PER_BLOCK = 1024;

    // ----------------------------------------------------------------
    // --- Vector field pathfinding algorithm CPU + CUDA
    /*
    Based on:
        - "How do vector field pathfinding algorithm work?" @ https://www.youtube.com/watch?v=ZJZu3zLMYAc
        - "How Flow Field Pathfinding Works - Flow Fields in Unity ep. 1" @ https://www.youtube.com/watch?v=zr6ObNVgytk&t=600s
        - "Vector-Flow-Field" @ https://github.com/Woestijnbok/Vector-Flow-Field
    */

    enum class FlowFieldDirection
    {
        NONE,
        UP,
        DOWN,
        LEFT,
        RIGHT,
        UP_LEFT,
        UP_RIGHT,
        DOWN_LEFT,
        DOWN_RIGHT
    };

    __global__ inline void generateFlowField(
        const Weight *__restrict__ integrationField,
        FlowFieldDirection *__restrict__ flowField,
        size_t rows,
        size_t columns,
        size_t totalCells)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= totalCells)
        {
            return;
        }

        const int directions[8][2] = {
            {-1, 0},  // UP
            {1, 0},   // DOWN
            {0, -1},  // LEFT
            {0, 1},   // RIGHT
            {-1, -1}, // UP_LEFT
            {-1, 1},  // UP_RIGHT
            {1, -1},  // DOWN_LEFT
            {1, 1}    // DOWN_RIGHT
        };

        Coordinates currentCoordinates = Graph::indexToCoordinates(idx, columns);
        FlowFieldDirection bestDirection = FlowFieldDirection::NONE;
        Weight minWeight = INFINITY;
        for (size_t i = 0; i < 8; ++i)
        {
            int neighborRow = currentCoordinates.row + directions[i][0];
            int neighborCol = currentCoordinates.column + directions[i][1];

            if (Graph::isValidDirection(neighborRow, neighborCol, rows, columns))
            {
                int neighborIdx = neighborRow * columns + neighborCol;
                Weight neighborWeight = integrationField[neighborIdx];
                if (neighborWeight < minWeight)
                {
                    minWeight = neighborWeight;
                    bestDirection = static_cast<FlowFieldDirection>(i + 1);
                }
            }
        }
        flowField[idx] = bestDirection;
    }

    [[nodiscard]] __host__ inline std::vector<VertexID> flowFieldPathfindingAccelerated(
        const Graph &graph,
        const PerlinNoiseGenerator &map,
        const Coordinates &start,
        const Coordinates &end,
        Timer &timer)
    {
        timer.reset();
        timer.resume();
        size_t mapRows = map.GRID_ROWS;
        size_t mapColumns = map.GRID_COLUMNS;
        size_t endRow = end.row;
        size_t endColumn = end.column;

        struct OpenListElement
        {
            size_t vertexRow;
            size_t vertexColumn;
            VertexID vertexID;
            Weight weightToGoal;

            OpenListElement(size_t row, size_t column, VertexID id, Weight distance)
                : vertexRow(row),
                  vertexColumn(column),
                  vertexID(id),
                  weightToGoal(distance) {}
        };

        std::queue<std::shared_ptr<OpenListElement>> openQueue;
        std::shared_ptr<OpenListElement> goal = std::make_shared<OpenListElement>(
            endRow,
            endColumn,
            Graph::coordinatesToIndex(endRow, endColumn, mapColumns),
            0);
        openQueue.push(goal);

        std::vector<std::shared_ptr<OpenListElement>> closedList;
        closedList.resize(graph.vertexCount);
        for (size_t vertex = 0; vertex < graph.vertexCount; ++vertex)
        {
            Coordinates coordinates = Graph::indexToCoordinates(vertex, mapColumns);
            if (coordinates.row == endRow && coordinates.column == endColumn)
            {
                closedList[vertex] = goal;
            }
            else
            {
                closedList[vertex] = std::make_shared<OpenListElement>(
                    coordinates.row,
                    coordinates.column,
                    vertex,
                    Graph::INFINITE_WEIGHT);
            }
        }

        // Generating the integration field.
        while (!openQueue.empty())
        {
            std::shared_ptr<OpenListElement> current = openQueue.front();
            openQueue.pop();
            size_t vertexIndex = current->vertexID;
            for (const auto &[directionRow, directionColumn] : Graph::PATHFINDING_GRID_DIRECTIONS)
            {
                int neighborRow = current->vertexRow + directionRow;
                int neighborColumn = current->vertexColumn + directionColumn;
                if (Graph::isValidDirection(neighborRow, neighborColumn, mapRows, mapColumns))
                {
                    size_t neighborIndex = Graph::coordinatesToIndex(neighborRow, neighborColumn, mapColumns);
                    Weight neighborWeight = map.grid[neighborRow][neighborColumn];
                    std::shared_ptr<OpenListElement> neighbor = closedList[neighborIndex];
                    if (neighborWeight + current->weightToGoal < neighbor->weightToGoal)
                    {
                        neighbor->weightToGoal = neighborWeight + current->weightToGoal;
                        openQueue.push(neighbor);
                    }
                }
            }
        }

        // Generating the flow field.
        Weight *integrationField;
        cudaMallocManaged(&integrationField, closedList.size() * sizeof(Weight));
        for (size_t vertex = 0; vertex < closedList.size(); ++vertex)
        {
            integrationField[vertex] = closedList[vertex]->weightToGoal;
        }

        FlowFieldDirection *flowField;
        cudaMallocManaged(&flowField, closedList.size() * sizeof(FlowFieldDirection));

        dim3 dimGrid((closedList.size() + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK, 1, 1);
        dim3 dimBlock(MAX_THREADS_PER_BLOCK, 1, 1);
        generateFlowField<<<dimGrid, dimBlock>>>(integrationField, flowField, mapRows, mapColumns, closedList.size());
        cudaDeviceSynchronize();

        // Reconstructing the path.
        Coordinates currentCoordinates = start;
        std::vector<VertexID> path;
        while (currentCoordinates.row != end.row && currentCoordinates.column != end.row)
        {
            size_t currentIndex = Graph::coordinatesToIndex(currentCoordinates.row, currentCoordinates.column, mapColumns);
            FlowFieldDirection currentDirection = flowField[currentIndex];
            path.push_back(currentIndex);
            switch (currentDirection)
            {
            case FlowFieldDirection::UP:
                currentCoordinates.row--;
                break;
            case FlowFieldDirection::DOWN:
                currentCoordinates.row++;
                break;
            case FlowFieldDirection::LEFT:
                currentCoordinates.column--;
                break;
            case FlowFieldDirection::RIGHT:
                currentCoordinates.column++;
                break;
            case FlowFieldDirection::UP_LEFT:
                currentCoordinates.row--;
                currentCoordinates.column--;
                break;
            case FlowFieldDirection::UP_RIGHT:
                currentCoordinates.row--;
                currentCoordinates.column++;
                break;
            case FlowFieldDirection::DOWN_LEFT:
                currentCoordinates.row++;
                currentCoordinates.column--;
                break;
            case FlowFieldDirection::DOWN_RIGHT:
                currentCoordinates.row++;
                currentCoordinates.column++;
                break;
            default:
                break;
            }
        }
        timer.pause();
        return path;
    }

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
        timer.reset();
        timer.resume();

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

        std::vector<VertexID> path = reconstructPath(predecessor, startingVertex, endingVertex);

        cudaFree(shortestDistances);
        cudaFree(updatingShortestDistances);
        cudaFree(activeVertices);
        cudaFree(d_adjacencyList);
        cudaFree(d_weights);
        cudaFree(d_edgesOffsets);
        cudaFree(d_edgesSize);

        timer.pause();
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
        timer.reset();
        timer.resume();

        size_t vertexCount = graph.vertexCount;
        std::vector<Weight> shortestDistances(vertexCount, Graph::INFINITE_WEIGHT);
        std::vector<VertexID> predecessor(vertexCount);
        std::vector<bool> visited(vertexCount, false);

        using QueueElement = std::pair<Weight, VertexID>;
        std::priority_queue<QueueElement, std::vector<QueueElement>, std::greater<QueueElement>> priorityQueue;

        shortestDistances[start] = 0;
        priorityQueue.push({0, start});

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
        std::vector<VertexID> path = reconstructPath(predecessor.data(), start, end);
        timer.pause();
        return path;
    }
}

#endif // PATHFINDING_CUH