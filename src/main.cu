#include "Graph.cuh"
#include "PerlinNoiseGenerator.cuh"
#include <iostream>

using namespace graph;

int main(void)
{
    // Graph graph(10);
    // graph.addConnection(0, 1, 1);
    // graph.addConnection(1, 2, 2);
    // graph.addConnection(2, 3, 3);
    // graph.addConnection(3, 4, 4);

    // std::cout << graph << std::endl;
    procedural_generation::PerlinNoiseGenerator::generateAndSave();
    return EXIT_SUCCESS;
}
