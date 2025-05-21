# Synthetic movement data generation using Mesa

## About Mesa

[Mesa](https://github.com/projectmesa/mesa) allows users to quickly create agent-based models using built-in
core components (such as spatial grids and agent schedulers) or
customized implementations; visualize them using a browser-based
interface; and analyze their results using Python's data analysis
tools. Its goal is to be the Python-based alternative to NetLogo,
Repast, or MASON.


## User guide

The generators are agent based models for movement in continuous space built using Mesa. 

To use the generators, follow the instructions in corresponding directory: 

1. [Bird Movement Model](./boid_flockers) (Mesa official example): 
   [Boids](https://en.wikipedia.org/wiki/Boids)-style flocking model, demonstrating the use of agents moving through a continuous space following direction vectors.
2. [Ship Movement Model](./ships_hybrid_algorithm) (MobiSpaces addition): 
   Agent based simulation of ship movements between ports, using a dynamic window approach and A* routing with support for speed restrictions.


## Acknowledgements

This work was supported in part by the Horizon Framework Programme of the European Union under grant agreement No. 101070279 ([MobiSpaces](https://mobispaces.eu)). 

