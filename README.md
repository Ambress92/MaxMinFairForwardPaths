# Beyond Shortest Paths: Node Fairness in Route Recommendations

<small>*Traditionally, route recommendation systems focused on minimizing distance (or time) for travelling between two points. However, recent attention has shifted to other factors beyond mere length. This paper addresses the challenge of ensuring a fair distribution of visits among network nodes when handling a high volume of point-to-point path queries. In doing so, we adopt a Rawlsian notion of individual-level fairness exploiting the power of randomness. Specifically, we aim to create a probabilistic distribution over paths that maximizes the minimum probability of any eligible node being included in the recommended path. A key idea of our work is the notion of forward paths, i.e., paths where traveling along any edge decreases the distance to the destination. In unweighted graphs forward paths and shortest paths coincide, but in weighted graphs forward paths provide a richer set of alternative routes, involving many more nodes while remaining close in length to the shortest path. Thus, they offer diversity and a wider basis for fairness. We devise an algorithm that extracts a directed acyclic graph (DAG) containing all the forward paths in the input graph, with the same computational runtime as solving a single shortest-path query. This avoids enumerating all possible forward paths, which can be exponential in the number of nodes. We then design a flow problem on this DAG to derive the probabilistic distribution over forward paths with the desired fairness property, solvable in polynomial time through a sequence of small linear programs. Our experiments on real-world datasets validate our theoretical results, demonstrating that our method provides individual node satisfaction while maintaining near-optimal path lengths. Moreover, our experiments show that our method can handle networks with millions nodes and edges on a commodity laptop and scales better than the baselines when there is a large volume of path queries for the same source and destination pair.*</small>

---

## Description

This repository contains the code to extracts the DAG of forward paths from a source to a target node of a graph, to obtain the Maxmin-fair distribution over the forward paths and to sample paths from it.

## Project Structure

- 📄 *MMFP.py*: main end to end script to run our method. It creates the DAG of forward paths, the Maxmin-fair distribution and to sample paths from it.

- 📄 *requirements.txt*: contains Python libraries requirements

- 📁 `src/`:  folder containg the main functions in *utils.py*

- 📁 `Notebooks/`:  folder which contains jupyter notebooks to reproduce the results of our experimental evaluation

- 📁 `Datasets/`: contains the datasets of the five cities (Piedmont, Essaoira, Florence, Buenos Aires and Kyoto) from Open Street Maps. 




## Requirements

Our code runs in Python 3 and uses the Linear Programming solver Gurobi, specifically:

``` 
Python 3.11.5
gurobipy 12.0.1
```

Additional details are contained in the 📄 *requirements.txt* file.


## Running our method

Our method can be simply run with the Python script 📄 *MMFP.py*:

Example:

``` 
python MMFP.py --graph Datasets/Piedmont__California__USA.pkl --orig_node 53123865 --dest_node 53075311
```

### Parameters
```
--graph: Path to the input graph file (pickled, Networkx DiGraph)
--orig_node: Source node 
--dest_node: Destination node 
--weight: Weight attribute in the graph (default: length)
--n_paths: Number of paths to sample (default: 100)
--seed: Random seed (default: 1)
```
### Output

1) A pickled file containing a Networkx DiGraph. The output graph corresponds to the DAG of forward paths from the source to the destination node. The DAG edge weights 'prob' and 'cond_prob' represent the absolute and conditioned transition probabilities corresponding to the Maxmin Fair Forward Path distribution. 

2) A text file containing a list of n_paths (from the source node to the destination node) sampled from the Maxmin Fair distribution.



## Datasets

The datasets of the cities from Open Street Maps are in the 📁 `Datasets/` folder (they can also be retrieved with the notebook 📄 *Save OSMnx datasets for MMFP.ipynb*). 

The dataset from the DIMACS shortest path challenge of the state of Florida and Eastern USA can be downloaded from: http://www.diag.uniroma1.it/challenge9/download.shtml

## Notebooks

- 📄 *MMFP path examples.ipynb*: Produces the DAG, the maxmin fair path distribution and examples of the paths for the city center of Florence, where a user needs to go from the Central Train Station to the Uffizi Gallery.
- 📄 *Paths draws.ipynb*: Draws the paths for our method and baselines methods on the DAG of forward paths for a source-target pair.
- 📄 *Resources usage.ipynb*: Notebook to compute the runtime and memory allocation of the methods
- 📄 *Save OSMnx datasets for MMFP.ipynb*: Retrieves and saves the datasets from Open Street Maps. 
- 📄 *Toy example.ipynb*: A toy example on synthetic data to show the MMFP method

## Competitors
To run the competitors the following repositories should be downloaded, placing them at the same level of this repository:

https://github.com/tchond/kspwlo/tree/master : to run ESX-C and OP+ algorithms.

https://github.com/AngelZihan/Diversified-Top-k-Route-Planning-in-Road-Network : to run the DKSP algorithm.

Furthermore, Yen's algorithm can be run in Networkx (see https://networkx.org/documentation/networkx-2.4/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html) with:
```
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
```




