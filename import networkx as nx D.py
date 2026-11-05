import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 4039
seeds = [1, 2, 3, 4, 5]  # 5 διαφορετικά seeds για κάθε πείραμα

# -----------------------------------------------------------------------------
# 1. ΠΕΙΡΑΜΑ ER: Αλλαγή του p (πιθανότητα σύνδεσης)
# -----------------------------------------------------------------------------
print("Εκτέλεση πειράματος για Erdős-Rényi...")
er_probs = [0.005, 0.0108, 0.015, 0.020]
er_results = []

for p in er_probs:
    edges_list, mean_k_list, max_k_list, min_k_list = [], [], [], []
    c_avg_list, c_glob_list, diam_list = [], [], []
    
    for s in seeds:
        G = nx.erdos_renyi_graph(n=N, p=p, seed=s)
        degs = [d for _, d in G.degree()]
        edges_list.append(G.number_of_edges())
        mean_k_list.append(np.mean(degs))
        max_k_list.append(np.max(degs))
        min_k_list.append(np.min(degs))
        c_avg_list.append(nx.average_clustering(G))
        c_glob_list.append(nx.transitivity(G))
        
        # Υπολογισμός διαμέτρου στη LCC
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        diam_list.append(nx.diameter(lcc))
        
    er_results.append({
        'param': p,
        'edges_m': np.mean(edges_list), 'edges_s': np.std(edges_list),
        'c_avg_m': np.mean(c_avg_list), 'c_avg_s': np.std(c_avg_list),
        'c_glob_m': np.mean(c_glob_list), 'c_glob_s': np.std(c_glob_list),
        'diam_m': np.mean(diam_list), 'diam_s': np.std(diam_list)
    })

df_er = pd.DataFrame(er_results)

# -----------------------------------------------------------------------------
# 2. ΠΕΙΡΑΜΑ WS: Αλλαγή του p (πιθανότητα επανασύνδεσης) με k=44 σταθερό
# -----------------------------------------------------------------------------
print("Εκτέλεση πειράματος για Watts-Strogatz...")
ws_probs = [0.001, 0.01, 0.05, 0.2]
ws_results = []

for p in ws_probs:
    c_avg_list, diam_list = [], []
    for s in seeds:
        G = nx.watts_strogatz_graph(n=N, k=44, p=p, seed=s)
        c_avg_list.append(nx.average_clustering(G))
        lcc = G.subgraph(max(nx.connected_components(G), key=len))
        diam_list.append(nx.diameter(lcc))
        
    ws_results.append({
        'param': p,
        'c_avg_m': np.mean(c_avg_list), 'c_avg_s': np.std(c_avg_list),
        'diam_m': np.mean(diam_list), 'diam_s': np.std(diam_list)
    })

df_ws = pd.DataFrame(ws_results)

# -----------------------------------------------------------------------------
# 3. ΣΧΕΔΙΑΣΗ ΔΙΑΓΡΑΜΜΑΤΩΝ ΜΕ ERROR BARS
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Γράφημα ER: Clustering vs p
ax1.errorbar(df_er['param'], df_er['c_avg_m'], yerr=df_er['c_avg_s'], fmt='-o', color='blue', capsize=4)
ax1.set_title("ER: Μέσο Clustering συναρτήσει του p")
ax1.set_xlabel("Πιθανότητα p")
ax1.set_ylabel("Average Clustering Coefficient")
ax1.grid(True)

# Γράφημα WS: Clustering & Διάμετρος συναρτήσει του p
ax2.errorbar(df_ws['param'], df_ws['c_avg_m'], yerr=df_ws['c_avg_s'], fmt='-s', color='green', label='Clustering', capsize=4)
ax2.set_xscale('log')
ax2.set_xlabel("Πιθανότητα Rewiring p (Log scale)")
ax2.set_ylabel("Clustering Coefficient", color='green')
ax2.grid(True)

ax2_twin = ax2.twinx()
ax2_twin.errorbar(df_ws['param'], df_ws['diam_m'], yerr=df_ws['diam_s'], fmt='-^', color='red', label='Διάμετρος', capsize=4)
ax2_twin.set_ylabel("Διάμετρος (LCC)", color='red')

plt.tight_layout()
plt.savefig("experiment_part_d_errorbars.png", dpi=200)
print("-> Τα πειραματικά διαγράμματα του Μέρους Δ αποθηκεύτηκαν στο: experiment_part_d_errorbars.png")