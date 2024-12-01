#ifndef GRAPH_CUH
#define GRAPH_CUH

#include "Constants.cuh"
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <vector>

namespace graph
{
    using Node = size_t;
    using Weight = uint;

    class Graph
    {
    public:
        // ----------------------------------------------------------------
        // --- Public class constants
        static constexpr Weight NO_CONNECTION = static_cast<Weight>(-1);
        static constexpr const char *NO_CONNECTION_STR = "INF";

        // ----------------------------------------------------------------
        // --- Public methods
        __host__ Graph(size_t nodesCount = 0) : m_nodesCount(nodesCount)
        {
            if (nodesCount == 0)
            {
                throw std::invalid_argument("nodesCount must be greater than 0.");
            }

            nodes = new Node[nodesCount];
            for (size_t i = 0; i < nodesCount; i++)
            {
                nodes[i] = i;
            }

            connections = new Weight[nodesCount * nodesCount];
            for (size_t i = 0; i < nodesCount * nodesCount; i++)
            {
                connections[i] = NO_CONNECTION;
            }
        }

        __host__ __device__ Graph(const Graph &other) = delete;

        __host__ __device__ Graph &operator=(const Graph &other) = delete;

        __host__ __device__ size_t nodesCount() const
        {
            return m_nodesCount;
        }

        __host__ __device__ void addConnection(Node nodeIdA, Node nodeIdB, Weight weight)
        {
            checkNode(nodeIdA);
            checkNode(nodeIdB);
            connections[nodeIdA * m_nodesCount + nodeIdB] = weight;
        }

        __host__ __device__ Weight getConnection(Node nodeIdA, Node nodeIdB) const
        {
            checkNode(nodeIdA);
            checkNode(nodeIdB);
            return connections[nodeIdA * m_nodesCount + nodeIdB];
        }

        __host__ __device__ bool areConnected(Node nodeIdA, Node nodeIdB) const
        {
            checkNode(nodeIdA);
            checkNode(nodeIdB);
            return connections[nodeIdA * m_nodesCount + nodeIdB] != NO_CONNECTION;
        }

        __host__ std::vector<Node> getNeighbors(Node nodeId) const
        {
            checkNode(nodeId);
            std::vector<Node> neighbors;
            for (size_t i = 0; i < m_nodesCount; i++)
            {
                if (areConnected(nodeId, i))
                {
                    neighbors.push_back(i);
                }
            }
            return neighbors;
        }

        __host__ ~Graph()
        {
            delete[] nodes;
            delete[] connections;
        }

        // ----------------------------------------------------------------
        // --- Public attributes
        Node *nodes;
        Weight *connections;

    private:
        // ----------------------------------------------------------------
        // --- Private class constants

        // ----------------------------------------------------------------
        // --- Private methods
        __host__ __device__ void checkNode(Node nodeId) const
        {
            assert(nodeId < m_nodesCount);
        }

        // ----------------------------------------------------------------
        // --- Private attributes
        size_t m_nodesCount;
    };

    __host__ inline std::ostream &operator<<(std::ostream &out, const Graph &graph)
    {
        const size_t WIDTH = 5;

        out << std::setw(WIDTH) << ' ';
        for (size_t i = 0; i < graph.nodesCount(); i++)
        {
            out << std::setw(WIDTH) << graph.nodes[i];
        }
        out << '\n';

        for (size_t i = 0; i < graph.nodesCount(); i++)
        {
            out << std::setw(WIDTH) << graph.nodes[i];
            for (size_t j = 0; j < graph.nodesCount(); j++)
            {
                if (graph.areConnected(i, j))
                {
                    out << std::setw(WIDTH) << graph.getConnection(i, j);
                }
                else
                {
                    out << std::setw(WIDTH) << Graph::NO_CONNECTION_STR;
                }
            }
            out << '\n';
        }
        return out;
    }

} // namespace graph

#endif // GRAPH_CUH