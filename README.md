# Beyond Shortest Paths: Node Fairness in Route Recommendations

[Antonio Ferrara](https://scholar.google.com/citations?user=v-tnmbwAAAAJ&hl=it) (CENTAI, Turin, Italy and TUGraz, Graz, Austria), [David Garcia Soriano](https://www.cs.upc.edu/~dgarcia/) (Universitat Politècnica de Catalunya
and Serra Húnter Fellow, Barcelona, Spain), and [Francesco Bonchi](https://www.francescobonchi.com) (CENTAI, Turin, Italy and Eurecat, Barcelona, Spain)

---

## Description

Repository of the paper "Beyond Shortest Paths: Node Fairness in Route Recommendations, 51st International Conference on Very Large Data Bases (VLDB), London, United Kingdom, 2025" (Link to be available soon). 

The repository contains the code to extract the Directed Acyclic Graph (DAG) of forward paths (paths in which each step along an edge brings a traveller closer to its destination) from a source to a destination node of a graph, to obtain the MaxMin-fair distribution over the Forward Paths (MMFP) and to sample paths from it.

The goal is to produce paths from a source to a target node in a way that the nodes that are on reasonable paths (forward paths) are probabilistically fairly visited according to the MaxMin-fair principle following the Rawlsian notion of individual-level fairness.

## Project Structure

- 📄 *MMFP.py*: main end to end script to run our method. It creates the DAG of forward paths, the Maxmin-fair distribution and samples paths from it.

- 📄 *requirements.txt*: contains Python libraries requirements.

- 📁 `src/`: folder containing the main functions in 📄 *utils.py*.

- 📁 `Notebooks/`:  folder which contains Jupyter Notebooks to reproduce the results of our experimental evaluation.

- 📁 `Datasets/`: contains the datasets of the five cities (Piedmont, Essaoira, Florence, Buenos Aires and Kyoto) from Open Street Maps. 

- 📁 `results/`: folder to store the results from 📄 *MMFP.py*.




## Requirements

Our code runs in Python 3 and uses the Linear Programming solver Gurobi, specifically:

``` 
Python 3.11.5
gurobipy 12.0.1
```

Additional details are contained in the 📄 *requirements.txt* file.


## How to run

Our method can be simply run with the Python script 📄 *MMFP.py*:

Example:

``` 
python MMFP.py --graph Datasets/Piedmont__California__USA.pkl --orig_node 53123865 --dest_node 53075311
```

### Parameters
```
--graph: Path to the input graph file (pickled, Networkx DiGraph)
--orig_node: Source node ID
--dest_node: Destination node ID
--weight: Weight attribute in the graph (default: length)
--n_paths: Number of paths to sample (default: 100)
--seed: Random seed (default: 1)
```
### Output

📄 *MMFP.py* saves two files in the 📁 `results/` folder:

1) A pickle file containing a Networkx DiGraph. The output graph corresponds to the DAG of forward paths from the source to the destination node. The DAG edge weights 'prob' and 'cond_prob' represent the absolute and conditioned transition probabilities corresponding to the Maxmin Fair Forward Path distribution. 

2) A text file containing a list of n_paths (from the source node to the destination node) sampled from the Maxmin Fair distribution.



## Datasets

The datasets of the cities from Open Street Maps are in the 📁 `Datasets/` folder (they can also be retrieved directly with the notebook 📄 *Save OSMnx datasets for MMFP.ipynb*). 

The dataset from the DIMACS shortest path challenge of the state of Florida and Eastern USA can be downloaded from: http://www.diag.uniroma1.it/challenge9/download.shtml . The downloaded datasets should be placed into the Dataset folder. They can then be transformed into Networkx Directed Graphs with the 📄 *Preprocess Dimacs datasets for MMFP.ipynb* short jupyter notebook.

## Notebooks

- 📄 *Lorenz Curves.ipynb*: Plots the generalized Lorenz curves for MMFP and competitors.
- 📄 *MMFP and competitors.ipynb*: Computes the average length, runtime and gini coefficient for our method and baseline methods.
- 📄 *MMFP path examples.ipynb*: Produces the DAG, the maxmin fair path distribution and examples of the paths for the city center of Florence, where a user needs to go from the Central Train Station to the Uffizi Gallery.
- 📄 *Paths draws.ipynb*: Draws the paths for our method and baseline methods on the DAG of forward paths for a source-target pair.
- 📄 *Preprocess Dimacs datasets for MMFP.ipynb*: Process the datasets from DIMACS shortest path challenge, converting them into Networkx DiGraphs.
- 📄 *Resources usage.ipynb*: Notebook to compute the runtime and memory allocation of the methods.
- 📄 *Save OSMnx datasets for MMFP.ipynb*: Retrieves and saves the datasets from Open Street Maps. 
- 📄 *Toy example.ipynb*: Toy example on synthetic data to show the MMFP method.

## Competitors
To run the competitors the following repositories should be downloaded, placing them at the same level of this repository:

https://github.com/tchond/kspwlo/tree/master : to run ESX-C and OP+ algorithms.

https://github.com/AngelZihan/Diversified-Top-k-Route-Planning-in-Road-Network : to run the DKSP algorithm.

Furthermore, Yen's algorithm can be run in Networkx (see https://networkx.org/documentation/networkx-2.4/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html) with:
```
def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
```




