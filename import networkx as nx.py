import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
import gzip
import shutil
import os

# -------------------------------------------------------------
# 1. Φόρτωση & Κατασκευή Δικτύων
# -------------------------------------------------------------
def load_or_download_facebook():
    filename = "facebook_combined.txt"
    if not os.path.exists(filename):
        gz_filename = "facebook_combined.txt.gz"
        url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
        print("Downloading Facebook dataset...")
        urllib.request.urlretrieve(url, gz_filename)
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    G = nx.read_edgelist(filename, nodetype=int, create_using=nx.Graph())
    return G

seed = 42
G_fb = load_or_download_facebook()
N = G_fb.number_of_nodes()
M = G_fb.number_of_edges()
avg_k = (2 * M) / N

# Δημιουργία Μοντέλων
p_er = avg_k / (N - 1)
G_er = nx.erdos_renyi_graph(n=N, p=p_er, seed=seed)

k_ws = int(round(avg_k))
if k_ws % 2 != 0: k_ws += 1
G_ws = nx.watts_strogatz_graph(n=N, k=k_ws, p=0.02, seed=seed)

m_ba = int(round(avg_k / 2))
G_ba = nx.barabasi_albert_graph(n=N, m=m_ba, seed=seed)

p_hk = 0.4
G_hk = nx.powerlaw_cluster_graph(n=N, m=m_ba, p=p_hk, seed=seed)

graphs = {
    "Facebook": G_fb,
    "Erdos-Renyi": G_er,
    "Watts-Strogatz": G_ws,
    "Barabasi-Albert": G_ba,
    "Holme-Kim": G_hk
}

# -------------------------------------------------------------
# 2. Υπολογισμός Μετρικών (Ερώτημα Α.α - Α.θ)
# -------------------------------------------------------------
results = {}

for name, G in graphs.items():
    print(f"Υπολογισμός μετρικών για: {name}")
    degrees = [d for _, d in G.degree()]
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Self-loops & Isolated nodes
    self_loops = nx.number_of_selfloops(G)
    isolated = nx.number_of_isolates(G)
    
    # Βαθμοί
    max_k = np.max(degrees)
    min_k = np.min(degrees)
    mean_k = np.mean(degrees)
    median_k = np.median(degrees)
    var_k = np.var(degrees)
    
    # Συνεκτικές συνιστώσες
    connected_components = list(nx.connected_components(G))
    num_cc = len(connected_components)
    largest_cc_nodes = max(connected_components, key=len)
    G_lcc = G.subgraph(largest_cc_nodes).copy()
    lcc_n = G_lcc.number_of_nodes()
    lcc_m = G_lcc.number_of_edges()
    
    # Αποστάσεις στη LCC
    diameter = nx.diameter(G_lcc)
    radius = nx.radius(G_lcc)
    avg_shortest_path = nx.average_shortest_path_length(G_lcc)
    
    # Clustering & Assortativity
    avg_clustering = nx.average_clustering(G)
    global_clustering = nx.transitivity(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    num_triangles = sum(nx.triangles(G).values()) // 3
    
    results[name] = {
        "Κόμβοι": num_nodes,
        "Ακμές": num_edges,
        "Self-loops": self_loops,
        "Isolated Nodes": isolated,
        "Max Degree": max_k,
        "Min Degree": min_k,
        "Mean Degree": f"{mean_k:.2f}",
        "Median Degree": median_k,
        "Degree Variance": f"{var_k:.2f}",
        "LCC Nodes": lcc_n,
        "LCC Edges": lcc_m,
        "Num CC": num_cc,
        "Diameter (LCC)": diameter,
        "Radius (LCC)": radius,
        "Avg Path Length (LCC)": f"{avg_shortest_path:.3f}",
        "Avg Clustering": f"{avg_clustering:.4f}",
        "Global Clustering": f"{global_clustering:.4f}",
        "Assortativity": f"{assortativity:.4f}",
        "Triangles": num_triangles
    }

df_results = pd.DataFrame(results).T
print(df_results)
df_results.to_csv("network_metrics_summary.csv")

# -------------------------------------------------------------
# 3. Σχεδίαση Διαγραμμάτων Κατανομής Βαθμών & C(k)
# -------------------------------------------------------------
def plot_distributions(graphs):
    for name, G in graphs.items():
        degrees = np.array([d for _, d in G.degree()])
        k_unique, counts = np.unique(degrees, return_counts=True)
        pk = counts / len(degrees)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"Ανάλυση Κατανομών - {name}", fontsize=14)
        
        # α) Γραμμικό ιστόγραμμα
        axes[0, 0].hist(degrees, bins=40, density=True, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("α) Linear Degree Histogram")
        axes[0, 0].set_xlabel("k")
        axes[0, 0].set_ylabel("P(k)")
        
        # β) Log-Log ιστόγραμμα
        axes[0, 1].scatter(k_unique, pk, alpha=0.7, color='darkblue')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].set_title("β) Log-Log Degree Scatter")
        axes[0, 1].set_xlabel("k")
        axes[0, 1].set_ylabel("P(k)")
        
        # γ) Log Binning
        bins = np.logspace(np.log10(max(1, degrees.min())), np.log10(degrees.max()), num=20)
        hist, bin_edges = np.histogram(degrees, bins=bins, density=True)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        valid = hist > 0
        axes[0, 2].plot(bin_centers[valid], hist[valid], 'o-', color='crimson')
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_title("γ) Log-Binning Histogram (Log-Log)")
        axes[0, 2].set_xlabel("k")
        axes[0, 2].set_ylabel("P(k)")
        
        # δ) Сumulative Degree Distribution (CCDF)
        sorted_deg = np.sort(degrees)
        ccdf = 1.0 - (np.arange(len(sorted_deg)) / float(len(sorted_deg)))
        axes[1, 0].plot(sorted_deg, ccdf, color='purple', lw=2)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_title("δ) Complementary CDF (Log-Log)")
        axes[1, 0].set_xlabel("k")
        axes[1, 0].set_ylabel("P(X >= k)")
        
        # ε) Zipf Rank-Frequency
        ranks = np.arange(1, len(sorted_deg) + 1)
        axes[1, 1].plot(ranks, sorted_deg[::-1], color='teal', lw=2)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title("ε) Zipf Plot (Rank vs Degree)")
        axes[1, 1].set_xlabel("Rank")
        axes[1, 1].set_ylabel("Degree k")
        
        # στ) C(k) vs k
        node_clustering = nx.clustering(G)
        k_to_c = {}
        for node, k in G.degree():
            k_to_c.setdefault(k, []).append(node_clustering[node])
        k_vals = np.array(sorted(k_to_c.keys()))
        avg_c = np.array([np.mean(k_to_c[k]) for k in k_vals])
        
        all_k = [G.degree(n) for n in G.nodes()]
        all_c = [node_clustering[n] for n in G.nodes()]
        axes[1, 2].scatter(all_k, all_c, alpha=0.15, color='gray', s=10, label='Nodes')
        axes[1, 2].plot(k_vals, avg_c, color='red', lw=2, label='Mean C(k)')
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        axes[1, 2].set_title("στ) Degree vs Clustering C(k)")
        axes[1, 2].set_xlabel("k")
        axes[1, 2].set_ylabel("C(k)")
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f"distributions_{name}.png", dpi=200)
        plt.close()

plot_distributions(graphs)