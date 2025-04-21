import graph_tool.collection 
import graph_tool.search 
import graph_tool as gt
import random
import numpy as np
import zstandard
import networkx as nx
import copy
import osmnx as ox
import random
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
from itertools import islice
import os
import subprocess
import ast
from itertools import combinations

#loading

def get_nx_from_ox(place = 'Piedmont, California, USA'):
    G = ox.graph_from_place(place, network_type='drive')
    return ox.utils_graph.convert.to_digraph(G, weight='length')

#sample sources and targets

def sample_st(g,k=100, seed = 42):
    random.seed(seed)

    nodes = list(g.nodes())

    sampled_src = random.choices(nodes, k=k)

    sampled_tar = random.choices(nodes, k=k)

    return [sampled_src,sampled_tar]


#funcs to get DAG of forward paths

def remove_edges_based_on_scores(graph_orig, dist_dict):
    graph = copy.deepcopy(graph_orig)
    edges_to_remove = []
    for node in graph.nodes():
        try:
            node_score = dist_dict[node]
            for neighbor in graph.neighbors(node):
                try:
                    neighbor_score = dist_dict[neighbor]
                except KeyError: 
                    neighbor_score = np.inf
                if neighbor_score >= node_score:
                    edges_to_remove.append((node, neighbor))
        except KeyError:
            continue
    for edge in edges_to_remove:
        graph.remove_edge(*edge)
    
    return graph


def neg_edge_weights(graph, wgt = 'length'):
    #inplace
    for source, target, data in graph.edges(data=True):
        if wgt not in data:
            raise ValueError(f"No weight attribute found for edge ({source}, {target})")
        data[wgt] = -data[wgt]


def remove_unreachable_nodes(G, source):
    # Get the set of nodes reachable from the source
    reachable_nodes = nx.descendants(G, source) | {source}

    # Find nodes that are not reachable
    non_reachable_nodes = set(G.nodes()) - reachable_nodes

    # Remove non-reachable nodes from the graph
    G.remove_nodes_from(non_reachable_nodes)
    
    return G


def get_dag(g, orig_node, dest_node, weight='length'):
    
    dist_dict = nx.shortest_path_length(g, source=None, target=dest_node, weight=weight, method='dijkstra')
    dag = remove_edges_based_on_scores(g, dist_dict)
    return remove_unreachable_nodes(dag,orig_node)



def get_dag_neg_weights(g, orig_node, dest_node, weight='length'):
    
    dist_dict = nx.shortest_path_length(g, source=None, target=dest_node, weight=weight, method='dijkstra')
    dag = remove_edges_based_on_scores(g, dist_dict)
    neg_edge_weights(dag, weight)
    return remove_unreachable_nodes(dag,orig_node)


#alternative implementation

def dag_fp(G: nx.DiGraph, s: int, t: int, weight) -> nx.DiGraph:
    """
    Creates a DAG containing the forward paths from source node s to target node t.
    
    Parameters:
    G (nx.DiGraph): The input directed graph.
    s (int): Source node.
    t (int): Target node.
    
    Returns:
    nx.DiGraph: A DAG with only forward paths from s to t.
    """
    # Compute shortest path distances from s and to t
    dist_from_s = nx.single_source_dijkstra_path_length(G, s, weight=weight)
    dist_to_t = nx.single_source_dijkstra_path_length(G.reverse(), t, weight=weight)
       
    # Find nodes that are reachable from s and can reach t
    reachable_from_s = set(dist_from_s.keys())
    reachable_to_t = set(dist_to_t.keys())
    reachable = reachable_from_s & reachable_to_t
    
    # Create an induced subgraph with only reachable nodes
    G_prime = G.subgraph(reachable).copy()
    
    # Remove edges that do not maintain the forward path condition
    for u, v in list(G_prime.edges()):
        if dist_to_t.get(u, float('inf')) <= dist_to_t.get(v, float('inf')):
            G_prime.remove_edge(u, v)
    
    return G_prime

#LP optim ONLY for worst - Later for all

def max_flow_with_alpha(G, s, t):
    # Initialize the model
    model = gp.Model('max_alpha_flow')

    # Create variables
    f = {}  # Flow variables
    for u, v in G.edges:
        f[u, v] = model.addVar(name=f"f_{u}_{v}", lb=0)

    alpha = model.addVar(name='alpha', lb=0)

    # Add constraints

    # alpha <= sum_u f(u, v) for all v != s, t
    for v in G.nodes:
        if v != s and v != t:
            model.addConstr(alpha <= gp.quicksum(f[u, v] for u in G.predecessors(v)), name=f"alpha_constr_{v}")

    # Flow conservation constraints: sum_u f(u, v) - sum_w f(v, w) = 0 for all v != s, t
    for v in G.nodes:
        if v != s and v != t:
            inflow = gp.quicksum(f[u, v] for u in G.predecessors(v))
            outflow = gp.quicksum(f[v, w] for w in G.successors(v))
            model.addConstr(inflow == outflow, name=f"flow_conserv_{v}")

    # sum_u f(u, t) = 1
    model.addConstr(gp.quicksum(f[u, t] for u in G.predecessors(t)) == 1, name="inflow_t")

    # sum_u f(s, u) = 1
    model.addConstr(gp.quicksum(f[s, u] for u in G.successors(s)) == 1, name="outflow_s")

    # Set the objective to maximize alpha
    model.setObjective(alpha, GRB.MAXIMIZE)

    # Optimize the model
    model.optimize()

    # Extract the results
    if model.status == GRB.OPTIMAL:
        result = {
            'alpha': alpha.X,
            'flow': { (u, v): f[u, v].X for u, v in f },
            'f': f
        }
    else:
        result = None

    return result






def compute_probabilities_and_expectations(dag, result, dest_node):
    flow = result['flow']
    
    # Initialize node attributes
    for node in dag.nodes():
        out_edges = list(dag.out_edges(node, data=True))
        total_out_flow = sum(flow[u, v] for u, v, _ in out_edges)
        
        dag.nodes[node]['prob'] = total_out_flow
        
        for u, v, data in out_edges:
            flow_value = flow[u, v]
            data['prob'] = flow_value
            
            if total_out_flow > 0:
                data['cond_prob'] = flow_value / total_out_flow
            else:
                data['cond_prob'] = 0
    
    # Ensure the probability of the target node is set to 1
    dag.nodes[dest_node]['prob'] = 1
    
    # Compute the expected length (edge weights) of a path
    edge_weights = np.array([data['length'] for u, v, data in dag.edges(data=True)])
    edge_probs = np.array([data['prob'] for u, v, data in dag.edges(data=True)])
    exp_length = (edge_weights * edge_probs).sum()
    
    # Compute the expected number of vertices visited in a path (including s and t)
    exp_numv = 1 + edge_probs.sum()
    
    return {
        
        "exp_length": exp_length,
        "exp_numv": exp_numv,
        "dag" : dag
    }


######### get k paths from LP and dag

def sample_forward_path(G, s, t):
    path = [s]
    current_node = s

    while current_node != t:
        next_nodes = list(G.successors(current_node))
        if not next_nodes:
            raise ValueError("No path from {} to {}".format(s, t))
        
        probabilities = [G[current_node][neighbor]['cond_prob'] for neighbor in next_nodes]
        next_node = random.choices(next_nodes, weights=probabilities)[0]
        path.append(next_node)
        current_node = next_node

    return path


def sample_random_fp(G, s, t):
    path = [s]
    current_node = s

    while current_node != t:
        next_nodes = list(G.successors(current_node))
        if not next_nodes:
            raise ValueError("No path from {} to {}".format(s, t))

        # Ignoring probabilities, assign equal weight to each next node
        weights = [1] * len(next_nodes)
        next_node = random.choices(next_nodes, weights=weights)[0]
        path.append(next_node)
        current_node = next_node

    return path


def sample_k_FP(final_dag, orig_node, dest_node,  num_paths):
    # Sample k paths from the LP distribution
    paths = []
    for _ in range(num_paths):
        paths.append(sample_forward_path(final_dag, orig_node, dest_node))

    return paths

def sample_k_random_FP(final_dag, orig_node, dest_node,  num_paths):
    # Sample k paths from a uniform (on the out-edges) distribution on the dag 
    paths = []
    for _ in range(num_paths):
        paths.append(sample_random_fp(final_dag, orig_node, dest_node))

    return paths


### k paths from yen algo

def yen_k_paths(G, source, target, weight, k):
     return list(
         islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
     )

#### k paths from kspwlo #####

def save_graph_to_gr(G, filename, source, target, weight):
    # Ensure the graph is directed
    if not G.is_directed():
        raise ValueError("The graph must be directed")
    
    # Create a mapping from original node IDs to new IDs starting from 0
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Get the number of nodes and edges
    num_nodes = len(node_mapping)
    
    # Prepare the data to be written
    data = []
    data.append("d")
    
    edges_data = []
    for u, v, attrs in G.edges(data=True):
        if weight in attrs:
            length = int(attrs[weight])
            new_u = node_mapping[u]
            new_v = node_mapping[v]
            edges_data.append(f"{new_u} {new_v} {length} 0")
    
    num_edges = len(edges_data)
    data.append(f"{num_nodes} {num_edges}")
    
    # Add edge information
    data.extend(edges_data)
    
    # Join the data into a single string with newlines
    data_str = "\n".join(data) + "\n"
    
    # Write to .gr file in text mode
    with open(filename, 'w') as file:
        file.write(data_str)
    
    
    #print(f"New source code: {new_source}")
    #print(f"New target code: {new_target}")
    
    return node_mapping


def save_graph_to_gr_EKSP(G, filename, source, target, weight):
    # Ensure the graph is directed
    if not G.is_directed():
        raise ValueError("The graph must be directed")

    # Create a mapping from original node IDs to new IDs starting from 1
    node_mapping = {node: i + 1 for i, node in enumerate(G.nodes())}
    # Get the number of nodes and edges
    num_nodes = len(node_mapping)
    
    # Prepare the header data (c for comment, p for problem line)
    data = []
    data.append("c This graph is converted from OpenStreetMap using OSMnx")
    data.append(f"p sp {num_nodes} {G.number_of_edges()}")  # `sp` for shortest path format (or other problem type)

    # Prepare the edges (a for arcs)
    edges_data = []
    for u, v, attrs in G.edges(data=True):
        if weight in attrs:
            length = int(attrs[weight])  # Edge weight (e.g., distance)
            new_u = node_mapping[u]      # New node id for u
            new_v = node_mapping[v]      # New node id for v
            edges_data.append(f"a {new_u} {new_v} {length}")
    
    # Add edge information to the data
    data.extend(edges_data)
    
    # Join the data into a single string with newlines
    data_str = "\n".join(data) + "\n"
    
    # Write to .gr file in text mode
    with open(filename, 'w') as file:
        file.write(data_str)
        
    return node_mapping


def execute_bash_command(file_path, k_value, t_value, s_value, d_value, a_value):
    # Construct the bash command using the provided parameters
    bash_command = f"./run.exec -f {file_path} -k {k_value} -t {t_value} -s {s_value} -d {d_value} -a {a_value}"

    try:
        # Execute the command and capture the output
        result = subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the command was successful
        if result.returncode == 0:
            # Decode the output from bytes to string
            output = result.stdout.decode('utf-8')
            return ast.literal_eval(output)  # Return the output
        else:
            # Return the error message if the command failed
            return f"Error executing the command:\n{result.stderr.decode('utf-8')}"
    
    except subprocess.CalledProcessError as e:
        # Return the error message if the command raised an exception
        return f"Command '{bash_command}' returned non-zero exit status {e.returncode}. Error:\n{e.stderr.decode('utf-8')}"


def parse_output(output):
    lines = output.strip().splitlines()
    
    path_data = None

    # Parse each line
    for line in lines:
        if line.startswith('[['):  # The path data starts here
            # Parse the list of lists, which can be evaluated as valid Python
            try:
                path_data = eval(line)  # Assuming it's safe since we know the structure
            except SyntaxError:
                print("Error parsing the path data.")
    
    return path_data

def execute_EKSP(file_path, k_value, sim_value):
    bash_command = f"../Diversified-Top-k-Route-Planning-in-Road-Network-main/DKSP {file_path} tmpQ.txt {k_value} {sim_value}"

    try:
        result = subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')

        # Parse the output and return only the path data (list of lists)
        path_data = parse_output(output)
        
        return path_data

    except subprocess.CalledProcessError as e:
        return f"Command '{bash_command}' returned non-zero exit status {e.returncode}. Error:\n{e.stderr.decode('utf-8')}"



### avg path length ###
def inverse_mapping(paths, node_map):
    """
    Transform the given paths using the inverse of the node_map.

    Parameters:
    paths (list of list of int): List of paths where each path is a list of node indices.
    node_map (dict): Mapping from original nodes to new nodes.

    Returns:
    list of list of int: Transformed paths with the new node indices.
    """
    # Create the inverse mapping
    inverse_node_map = {v: k for k, v in node_map.items()}
    
    # Transform each path using the inverse mapping
    transformed_paths = []
    if paths:  # Check if paths is not None and not empty
        for path in paths:
            transformed_path = [inverse_node_map[node] for node in path]
            transformed_paths.append(transformed_path)
    return transformed_paths


def increase_values_by_one(node_map):
    # Create a new dictionary with the same keys and values increased by 1
    new_node_map = {key: value + 1 for key, value in node_map.items()}
    return new_node_map

def average_path_length(paths, g, weight):
    """
    Calculate the average length of the paths in the graph g using the weight attribute wgt.

    Parameters:
    paths (list of list of int): List of paths where each path is a list of node indices.
    g (networkx.DiGraph): The directed graph with weighted edges.
    weight (str): The attribute name for the edge weights.

    Returns:
    float: The average length of the paths.
    """
    
    total_length = 0
    valid_paths_count = 0
    
    for path in paths:
        path_length = 0
        try:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_length += g[u][v][weight]
            total_length += path_length
            valid_paths_count += 1
        except KeyError:
            # If a path has nodes with no connecting edge or if the weight attribute is missing,
            # we skip this path and continue with the others.
            print(f"Skipping invalid path: {path}")
            continue
    
    if valid_paths_count == 0:
        return 0  # To avoid division by zero if no valid paths are found

    return total_length / valid_paths_count




### Diversity ####

# Define a function to calculate symmetric difference length between two paths
def symmetric_difference_nodes(path1, path2):
    set1, set2 = set(path1), set(path2)
    symmetric_diff = set1.symmetric_difference(set2)
    return len(symmetric_diff)


def mean_symmetric_difference_nodes(paths):

    # Generate all pairs of paths
    path_pairs = list(combinations(paths, 2))

    # Calculate the symmetric difference lengths for all pairs
    symmetric_diff_lengths = [
        symmetric_difference_nodes(path1, path2) for path1, path2 in path_pairs
    ]

    # Calculate and return the mean of the symmetric difference lengths
    return np.mean(symmetric_diff_lengths)


#deviding by the longest

def normalized_symmetric_difference_nodes(path1, path2):
    set1, set2 = set(path1), set(path2)
    symmetric_diff = set1.symmetric_difference(set2)
    larger_set_size = max(len(set1), len(set2))
    return len(symmetric_diff) / larger_set_size

def mean_normalized_symmetric_difference_nodes(paths):
    # Generate all pairs of paths
    path_pairs = list(combinations(paths, 2))

    # Calculate the normalized symmetric difference lengths for all pairs
    normalized_diff_lengths = [
        normalized_symmetric_difference_nodes(path1, path2) for path1, path2 in path_pairs
    ]

    # Calculate and return the mean of the normalized symmetric difference lengths
    return np.mean(normalized_diff_lengths)


def jaccard_diversity(path1, path2):
    set1, set2 = set(path1), set(path2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0  # Avoid division by zero if both sets are empty
    return 1-len(intersection) / len(union)

def mean_jaccard_diversity(paths):
    # Generate all pairs of paths
    path_pairs = list(combinations(paths, 2))

    # Calculate the normalized symmetric difference lengths for all pairs
    jacs = [
        jaccard_diversity(path1, path2) for path1, path2 in path_pairs
    ]

    # Calculate and return the mean of the normalized symmetric difference lengths
    return np.mean(jacs)

## counting unique paths in a list of paths

def count_unique_paths(paths):
    # Convert each path to a tuple
    paths_as_tuples = [tuple(path) for path in paths]
    # Use a set to filter out duplicate paths
    unique_paths = set(paths_as_tuples)
    # Return the number of unique paths
    return len(unique_paths)


def remove_zero_prob_edges(graph):
    # Create a deep copy of the original graph to avoid modifying it
    new_graph = copy.deepcopy(graph)
    
    # Collect edges with 'cone_prob' equal to 0
    edges_to_remove = [(u, v) for u, v, data in new_graph.edges(data=True) if data.get('cond_prob', 0) == 0]
    
    # Remove the edges with zero 'cone_prob'
    new_graph.remove_edges_from(edges_to_remove)
    
    return new_graph


##### filter source - target

def filter_st(g, wgt, st):

    # Resultant lists after filtering
    filtered_sources = []
    filtered_destinations = []
    
    for src, dest in zip(st[0], st[1]):
        try:
            # Compute the shortest path using Dijkstra's algorithm
            shortest_path = nx.dijkstra_path(g, src, dest, weight=wgt)
            # Check the number of nodes in the shortest path
            if len(shortest_path) > 6:
                filtered_sources.append(src)
                filtered_destinations.append(dest)
        except nx.NetworkXNoPath:
            # If there is no path between src and dest
            continue
    
    return [filtered_sources, filtered_destinations]


#### list of shortest path length and list of number of nodes for st pairs

def list_shortest_path_nx(g, weights, sampled_indices_src, sampled_indices_tar):
    lst = []
    num_nodes_sp = []
    inf_counter = 0
    for src, tar in zip(sampled_indices_src, sampled_indices_tar):
        if nx.has_path(g, src, tar):
            shortest_path_length, shortest_path = nx.single_source_dijkstra(g, source=src, target=tar, weight=weights)
            lst.append(shortest_path_length)
            num_nodes_sp.append(len(shortest_path))
        else:
            inf_counter += 1 
            print(src, tar)
        
    print('unreachable targets:', inf_counter)    
    return lst,num_nodes_sp

#### list of LFP length, LFP nodes, DAG nodes

def list_longest_forward_path(g, wgt, sampled_indices_src, sampled_indices_tar):

    lst = []
    num_nodes = []
    tot_nodes = []
    inf_counter = 0

    for i in range(len(sampled_indices_src)):

        dag = get_dag_neg_weights(g, sampled_indices_src[i], sampled_indices_tar[i], weight=wgt)
        try:
                dist,path = nx.single_source_dijkstra(dag, source=sampled_indices_src[i], target=sampled_indices_tar[i], weight=wgt)
                lst.append(-dist)
                num_nodes.append(len(path))
                tot_nodes.append(len(dag.nodes()) )

        except:
            inf_counter +=1
            print(sampled_indices_src[i], sampled_indices_tar[i])
            print('unreachable targets:', inf_counter)
    
    return lst,num_nodes,tot_nodes


#####solving full maxmin fairness

def solve_primal(G, s, t, K, alpha, track_memory = False):
    # Initialize the primal model
    primal_model = gp.Model('primal_max_lambda_flow')
    primal_model.setParam('OutputFlag', 0)

    # Create primal variables
    f = {}  # Flow variables
    for u, v in G.edges:
        f[u, v] = primal_model.addVar(name=f"f_{u}_{v}", lb=0)

    lambda_ = primal_model.addVar(name='lambda', lb=0)
    
    primal_model.update()
    
    # Add primal constraints using addConstrs
    alpha_cons = primal_model.addConstrs(
        (-gp.quicksum(f[u, v] for u in G.predecessors(v)) <= -alpha[v])
        for v in G.nodes if v != s and v in K
    )

    lambda_cons = primal_model.addConstrs(
        (lambda_ <= gp.quicksum(f[u, v] for u in G.predecessors(v)))
        for v in G.nodes if v != s and v not in K
    )

    flow_cons = primal_model.addConstrs(
        (gp.quicksum(f[u, v] for u in G.predecessors(v)) == gp.quicksum(f[v, w] for w in G.successors(v)))
        for v in G.nodes if v != s and v != t
    )

    # Add specific constraints for s and t
    primal_model.addConstr(gp.quicksum(f[u, t] for u in G.predecessors(t)) == 1)
    primal_model.addConstr(gp.quicksum(f[s, u] for u in G.successors(s)) == 1)
    
    primal_model.update()
    
    # Set the objective to maximize lambda_
    primal_model.setObjective(lambda_, GRB.MAXIMIZE)

    
    if track_memory:
        # Variable to store peak memory usage
        peak_memory_usage = [0]# Using a mutable list to store value inside the callback
        memory_usage_2 = [0]

        def mycallback(model, where):
            if where == GRB.Callback.SIMPLEX:
                mem_used = model.cbGet(GRB.Callback.MAXMEMUSED)
                peak_memory_usage[0] = max(peak_memory_usage[0], mem_used)  # Track max memory used            
                mem_used_2 = model.cbGet(GRB.Callback.MEMUSED)
                memory_usage_2[0] = max(memory_usage_2[0], mem_used_2)  # Track max memory used
                #print(f"Memory used: {mem_used:.2f} MB")

        # Optimize the primal model
        primal_model.optimize(mycallback)

        # Get peak memory usage (in MB) from the stored value
        peak_memory = peak_memory_usage[0]
        memory_used = memory_usage_2[0]
        #print('Peak memory usage:', peak_memory)
        #print('Memory used:', memory_used)
        
        return primal_model, f , (flow_cons,alpha_cons,lambda_cons),peak_memory,memory_used
        
    else:    

        # Optimize the primal model
        primal_model.optimize()



        return primal_model, f , (flow_cons,alpha_cons,lambda_cons), -1,-1


def solve_dual_from_primal(dag, model, all_vars, all_cons, s, t, K):    
    
    fairness = model.objVal
    f = all_vars
    flow_cons, alpha_cons, lambda_cons = all_cons

    # Get the dual weights
    w = {}
    d = {}
    for (v, cons) in alpha_cons.items():
        w[v] = cons.Pi                     #### is it cons or -cons? in your code was -cons
    for (v, cons) in lambda_cons.items():
        w[v] = cons.Pi                     ##### apparently here can not be -cons.Pi or it doesn't run
    for (v, cons) in flow_cons.items():
        d[v] = cons.Pi
    d[t] = fairness
    
    
    # Compute the sum of w[v] for all v
    w_sum = sum(w[v] for v in w if v not in K)
    
    # Ensure the sum of weights for nodes in K is close to 1
    assert abs(w_sum - 1) < 1e-6, f"Sum of weights for nodes in K is {w_sum}, expected 1"

    
    return w,d

def iterative_solver(G, s, t, printing= False, track_memory = False):
    K = set()   # Initialize K as an empty set
    alpha = {}  # Initialize alpha dictionary
    
    peak_memory = -1
    memory_used = -1
    
    # Repeat until K includes all nodes in the graph
    while len(K) < len(G.nodes)-1:
        # Solve the primal LP and its dual
        model, all_vars, all_cons, curr_peak_memory, curr_memory_used = solve_primal(G, s, t, K, alpha, track_memory)
        
        peak_memory = max(peak_memory,curr_peak_memory)
        memory_used = max(memory_used,curr_memory_used)
        
        w, d = solve_dual_from_primal(G, model, all_vars, all_cons, s, t, K)
        
        # Update K with vertices where w_v > 0
        K_prime = {v for v in G.nodes if v not in K and w.get(v, 0) > 0}
        K.update(K_prime)
        
        # Update alpha for the new vertices in K
        for v in K_prime:
            alpha[v] = model.objVal  # lambda^* value
        
        if printing:
            print(len(G.nodes)-1 - len(K))

    alpha[s]= 1   
        
    result = {
            
            'flow': { (u, v): all_vars[u, v].X for u, v in all_vars }
        }
    
    if track_memory:
        return K, alpha, model, result, peak_memory*1000, memory_used*1000 #memory in MB this way
    else:
    
        return K, alpha, model, result


#### add nodes satisfaction to dag for method

def add_nodes_satisfaction(paths, g, method_name):
    #inplace

    num_paths = len(paths)
    # Count node appearances
    node_counts = defaultdict(int)
    for path in paths:
        for node in path:
            node_counts[node] += 1

    # Normalize counts to get fp_sampled_flow values
    sampled_satis = {node: count / num_paths for node, count in node_counts.items()}

    # Add fp_sampled_flow to graph g
    for node in g.nodes:
        g.nodes[node][method_name] = sampled_satis.get(node, 0)



def get_gini(dag, attribute):
        flow_values = np.array([data[attribute] for node, data in dag.nodes(data=True)])
        
        if len(flow_values) == 0:
            raise ValueError(f"No non-zero values found for attribute '{attribute}'")
        
        # Step 1: Sort the flow values in ascending order
        sorted_flow_values = np.sort(flow_values)
        
        # Step 2: Compute the cumulative sum of the sorted flow values
        cumulative_sum = np.cumsum(sorted_flow_values)
        
        # Step 3: Normalize the cumulative sum
        total_sum = cumulative_sum[-1]
        lorenz_curve = cumulative_sum / total_sum
        
        # Compute the Gini index
        gini_index = compute_gini_index(lorenz_curve)

        return gini_index

def compute_gini_index(lorenz_curve):
    n = len(lorenz_curve)
    B = np.sum(lorenz_curve)
    Gini = 1 - 2 * (B - 0.5) / n
    return Gini

def get_gini_from_alpha(alpha):
        flow_values = np.array(list(alpha.values()))
        
        
        # Step 1: Sort the flow values in ascending order
        sorted_flow_values = np.sort(flow_values)
        
        # Step 2: Compute the cumulative sum of the sorted flow values
        cumulative_sum = np.cumsum(sorted_flow_values)
        
        # Step 3: Normalize the cumulative sum
        total_sum = cumulative_sum[-1]
        lorenz_curve = cumulative_sum / total_sum
        
        # Compute the Gini index
        gini_index = compute_gini_index(lorenz_curve)

        return gini_index


def compute_and_assign_node_flow(S, G, result_flow, source_node):
    """
    Compute the incoming flow for each node in subgraph S and assign it as a node attribute in graph G.
    Nodes in G that are not in S will have a node_flow value of 0.
    The source node will have its flow directly set to 1.
    
    Parameters:
    S (networkx.DiGraph): Directed subgraph. (The DAG)
    G (networkx.DiGraph): Full Directed graph.
    result_flow (dict): Dictionary with edges as keys and flow values as values.
    source_node (int): The source node which should have its flow set directly to 1.
    
    Returns:
    None: The function modifies graph G in place by adding node_flow attribute to its nodes.
    """
    
    # Step 1: Compute incoming flow for each node in S
    incoming_flow = {node: 0 for node in S.nodes()}

    for (src, dst), flow in result_flow.items():
        if dst in S.nodes():
            incoming_flow[dst] += flow

    # Step 2: Set the source node's flow directly to 1
    if source_node in S.nodes():
        incoming_flow[source_node] = 1

    # Step 3: Assign node flow to nodes in G
    node_flow = {node: incoming_flow[node] if node in incoming_flow else 0 for node in G.nodes()}

    # Add the node_flow as a node attribute in G
    nx.set_node_attributes(G, node_flow, 'node_flow')



