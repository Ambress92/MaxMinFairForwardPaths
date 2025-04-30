import argparse
import os
import pickle
import os.path
import numpy as np

from src import utils as ut

# example:
# python MMFP.py --graph Datasets/Piedmont__California__USA.pkl --orig_node 53123865 --dest_node 53075311 

def main():
    parser = argparse.ArgumentParser(description="Run LP solver and sample fair FPs on a DAG.")
    parser.add_argument('--graph', type=str, required=True, help='Path to the input graph file (pickled, Networkx DiGraph)')
    parser.add_argument('--orig_node', type=int, required=True, help='Origin node')
    parser.add_argument('--dest_node', type=int, required=True, help='Destination node')
    parser.add_argument('--weight', type=str, default='length', help='Weight attribute in the graph (default: length)')
    parser.add_argument('--n_paths', type=int, default=100, help='Number of paths to sample (default: 100)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')

    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.graph, 'rb') as f:
        g = pickle.load(f)

    graph_name = os.path.splitext(os.path.basename(args.graph))[0]

    # Compute DAG and solve
    dag = ut.get_dag(g, args.orig_node, args.dest_node, weight=args.weight)
    K, alpha, model, result = ut.iterative_solver(dag, args.orig_node, args.dest_node)
    final = ut.compute_probabilities_and_expectations(dag, result, args.dest_node)
    LP_dag = final['dag']
    fair_fp = ut.sample_k_FP(LP_dag, args.orig_node, args.dest_node, args.n_paths)

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # Construct filename suffix
    suffix = f"_{graph_name}_orig{str(args.orig_node)}_dest{str(args.dest_node)}"

    # Save the DAG with the Maxmin Fair transition probabilities (in a pickle file)
    with open(f"results/LP_dag{suffix}.pkl", "wb") as f:
        pickle.dump(LP_dag, f)

    print(f"LP_dag{suffix}.pkl saved in the results folder")

    # Save a sample of the Maxmin Fair forward paths (in a text file)
    with open(f"results/paths{suffix}.txt", "w") as f:
        for path in fair_fp:
            f.write(str(path) + "\n")

    print(f"paths{suffix}.txt saved in the results folder")


if __name__ == '__main__':
    main()

    
