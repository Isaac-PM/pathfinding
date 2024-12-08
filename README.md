# Competing Pathfinding Algorithms

<img src="cover.webp" width="600px">

> OpenAI. (2024). A visually dynamic illustration of a competition between pathfinding algorithms.

## Overview

The current project is a collection of three different pathfinding algorithms, two of them accelerated or implemented on top of CUDA, this was created as an assignment in the course "CUDA at Scale for the Enterprise", the assignment was to create a CUDA "game" where two agents compete using different GPUs, in my case I decided to create a pathfinding competition. The main idea is that two agents choose one of the algorithms (strategy) via exploration and exploration analysis and then solve a pathfinding problem on a grid, the grid (or map) is a 2D matrix created via a Perlin noise generator, the agents have to find the shortest path from the start point to the end point, the project workflow is as follows:

<img src="diagram.png" width="600px">

First, the game benchmarks each of the available pathfinding algorithms, which are:

- FFP: Flow Field Pathfinding (accelerated with CUDA)
- DCP: Dijkstra's algorithm (CPU)
- DCU: Dijkstra's Algorithm (CUDA)

All algorithms return the same result, as Flow Field is basically the same as Dijkstra's algorithm, the only difference is that Flow Field computes only the shortest path between the start and end point, while Dijkstra's algorithm computes the shortest path between all nodes in the graph. 

At the beginning, each algorithm is benchmarked using the best available GPU and classified into best, average, and worst according to the time taken by each algorithm. Usually DCP performs the best, this is explained by comments on the code, followed by FFP, and DCU which performs the worst, the reasons for DCU's performance can be further studied in the attached paper[^1]; this behavior is expected according to the O-complexity of each.

Then the game calls for a graph in each round, the graph is generated using a grid of size 1'000 * 1'000, resulting in 1'000'000 nodes with at least 4 neighbors, then a random start and end point is determined and the round is set for the agents to play.

The agents will choose an strategy (best, average or worst performing algorithm) based on the state of the game, they will be reward points according to the algorithm chosen:

| Strategy |      Points granted      |
|:--------:|:------------------------:|
| Best     | +10                      |
| Average  | +20                      |
| Worst    | +30 - (time taken / 100) |

Once the points are awarded, the round results are stored and the game continues until a certain number of rounds have been played. When the game is finished, the results can be visualized using a Streamlit application, which displays the game results, the path found by the agents, the time taken by each algorithm, and other relevant information.

## Building and Running the Project

## References

[^1]: Martín, P. J., Torres, R., & Gavilanes, A. (2009). CUDA solutions for the SSSP problem. In Lecture notes in computer science (pp. 904–913). https://doi.org/10.1007/978-3-642-01970-8_91

[^2]: Vitality. (n.d.). Dijkstra’s algorithm in CUDA. Stack Overflow. https://stackoverflow.com/questions/12763991/dijkstras-algorithm-in-cuda

[^3]: PDN - PasDeNom. (2020, October 25). How do vector field Pathfinding algorithm work ? [Video]. YouTube. https://www.youtube.com/watch?v=ZJZu3zLMYAc

[^4]: Turbo Makes Games. (2020, August 4). How Flow Field Pathfinding Works - Flow Fields in Unity ep. 1 [Video]. YouTube. https://www.youtube.com/watch?v=zr6ObNVgytk

[^5]: Woestijnbok. (n.d.). GitHub - Woestijnbok/Vector-Flow-Field: This project showcases and explains the upsides and downsides of the vector flow field pathfinding algorithm. GitHub. https://github.com/Woestijnbok/Vector-Flow-Field