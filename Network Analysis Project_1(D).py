import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# ΒΗΜΑ 1: ΒΑΣΙΚΕΣ ΡΥΘΜΙΣΕΙΣ
# =============================================================================
N = 4039                  # Σταθερό πλήθος κόμβων (όπως ζητάει η εκφώνηση)
SEEDS = [42, 67, 100, 264, 305]   # 5 διαφορετικά seeds για τις 5 επαναλήψεις

# Λίστα στην οποία θα μαζέψουμε όλα τα αποτελέσματα για να τα κάνουμε πίνακα
results_list = []

# =============================================================================
# ΒΗΜΑ 2: ΠΕΙΡΑΜΑ ΓΙΑ ERDŐS-RÉNYI (ER)
# Κύρια παράμετρος: Πιθανότητα σύνδεσης p
# =============================================================================

# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html
print("1/4. Εκτέλεση πειράματος για Erdős-Rényi (ER)...")
er_p_values = [0.005, 0.01082, 0.018, 0.025]

for p in er_p_values:
    print(f"   -> Υπολογισμός ER για p = {p} (5 επαναλήψεις)...")
    edges_runs, min_k_runs, max_k_runs, mean_k_runs = [], [], [], []
    diam_runs, c_avg_runs, c_glob_runs = [], [], []
    
    for s in SEEDS:
        # Φτιάχνουμε το δίκτυο ER
        G = nx.erdos_renyi_graph(n=N, p=p, seed=s)
        
        # Μετράμε ακμές
        edges_runs.append(G.number_of_edges())
        # Μετράμε βαθμούς
        degs = [d for _, d in G.degree()]
        min_k_runs.append(np.min(degs))
        max_k_runs.append(np.max(degs))
        mean_k_runs.append(np.mean(degs))
        # Μετράμε clustering
        c_avg_runs.append(nx.average_clustering(G))
        c_glob_runs.append(nx.transitivity(G))
        # Μετράμε διάμετρο στη μεγαλύτερη συνιστώσα (LCC)
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        diam_runs.append(nx.diameter(lcc))
        
    # Αποθηκεύουμε Μέση Τιμή (mean) και Τυπική Απόκλιση (std)
    results_list.append({
        "Μοντέλο": "ER",
        "Παράμετρος": p,
        "Edges_mean": np.mean(edges_runs), "Edges_std": np.std(edges_runs),
        "Min_k_mean": np.mean(min_k_runs), "Min_k_std": np.std(min_k_runs),
        "Max_k_mean": np.mean(max_k_runs), "Max_k_std": np.std(max_k_runs),
        "Mean_k_mean": np.mean(mean_k_runs), "Mean_k_std": np.std(mean_k_runs),
        "Diam_mean": np.mean(diam_runs), "Diam_std": np.std(diam_runs),
        "C_avg_mean": np.mean(c_avg_runs), "C_avg_std": np.std(c_avg_runs),
        "C_glob_mean": np.mean(c_glob_runs), "C_glob_std": np.std(c_glob_runs),
    })

# =============================================================================
# ΒΗΜΑ 3: ΠΕΙΡΑΜΑ ΓΙΑ WATTS-STROGATZ (WS)
# Κύρια παράμετρος: Πιθανότητα επανασύνδεσης p (το k=44 μένει σταθερό)
# =============================================================================

# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html
print("2/4. Εκτέλεση πειράματος για Watts-Strogatz (WS)...")
ws_p_values = [0.001, 0.01, 0.02, 0.1]

for p in ws_p_values:
    print(f"   -> Υπολογισμός WS για p = {p} (5 επαναλήψεις)...")
    edges_runs, min_k_runs, max_k_runs, mean_k_runs = [], [], [], []
    diam_runs, c_avg_runs, c_glob_runs = [], [], []
    
    for s in SEEDS:
        # Φτιάχνουμε το δίκτυο WS με σταθερό k=44
        G = nx.watts_strogatz_graph(n=N, k=44, p=p, seed=s)
        
        edges_runs.append(G.number_of_edges())
        degs = [d for _, d in G.degree()]
        min_k_runs.append(np.min(degs))
        max_k_runs.append(np.max(degs))
        mean_k_runs.append(np.mean(degs))
        c_avg_runs.append(nx.average_clustering(G))
        c_glob_runs.append(nx.transitivity(G))
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        diam_runs.append(nx.diameter(lcc))
        
    results_list.append({
        "Μοντέλο": "WS",
        "Παράμετρος": p,
        "Edges_mean": np.mean(edges_runs), "Edges_std": np.std(edges_runs),
        "Min_k_mean": np.mean(min_k_runs), "Min_k_std": np.std(min_k_runs),
        "Max_k_mean": np.mean(max_k_runs), "Max_k_std": np.std(max_k_runs),
        "Mean_k_mean": np.mean(mean_k_runs), "Mean_k_std": np.std(mean_k_runs),
        "Diam_mean": np.mean(diam_runs), "Diam_std": np.std(diam_runs),
        "C_avg_mean": np.mean(c_avg_runs), "C_avg_std": np.std(c_avg_runs),
        "C_glob_mean": np.mean(c_glob_runs), "C_glob_std": np.std(c_glob_runs),
    })

# =============================================================================
# ΒΗΜΑ 4: ΠΕΙΡΑΜΑ ΓΙΑ BARABÁSI-ALBERT (BA)
# Κύρια παράμετρος: Αριθμός ακμών m ανά νέο κόμβο
# =============================================================================

# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html
print("3/4. Εκτέλεση πειράματος για Barabási-Albert (BA)...")
ba_m_values = [10, 16, 22, 30]

for m in ba_m_values:
    print(f"   -> Υπολογισμός BA για m = {m} ...")
    edges_runs, min_k_runs, max_k_runs, mean_k_runs = [], [], [], []
    diam_runs, c_avg_runs, c_glob_runs = [], [], []
    
    for s in SEEDS:
        # Φτιάχνουμε το δίκτυο BA
        G = nx.barabasi_albert_graph(n=N, m=m, seed=s)
        
        edges_runs.append(G.number_of_edges())
        degs = [d for _, d in G.degree()]
        min_k_runs.append(np.min(degs))
        max_k_runs.append(np.max(degs))
        mean_k_runs.append(np.mean(degs))
        c_avg_runs.append(nx.average_clustering(G))
        c_glob_runs.append(nx.transitivity(G))
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        diam_runs.append(nx.diameter(lcc))
        
    results_list.append({
        "Μοντέλο": "BA",
        "Παράμετρος": m,
        "Edges_mean": np.mean(edges_runs), "Edges_std": np.std(edges_runs),
        "Min_k_mean": np.mean(min_k_runs), "Min_k_std": np.std(min_k_runs),
        "Max_k_mean": np.mean(max_k_runs), "Max_k_std": np.std(max_k_runs),
        "Mean_k_mean": np.mean(mean_k_runs), "Mean_k_std": np.std(mean_k_runs),
        "Diam_mean": np.mean(diam_runs), "Diam_std": np.std(diam_runs),
        "C_avg_mean": np.mean(c_avg_runs), "C_avg_std": np.std(c_avg_runs),
        "C_glob_mean": np.mean(c_glob_runs), "C_glob_std": np.std(c_glob_runs),
    })

# =============================================================================
# ΒΗΜΑ 5: ΠΕΙΡΑΜΑ ΓΙΑ HOLME-KIM (HK)
# Κύρια παράμετρος: Πιθανότητα κλεισίματος τριγώνου p (το m=22 μένει σταθερό)
# =============================================================================

# https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.powerlaw_cluster_graph.html
print("4/4. Εκτέλεση πειράματος για Holme-Kim (HK)...")
hk_p_values = [0.1, 0.25, 0.4, 0.7]

for p in hk_p_values:
    print(f"   -> Υπολογισμός HK για p = {p} ...")
    edges_runs, min_k_runs, max_k_runs, mean_k_runs = [], [], [], []
    diam_runs, c_avg_runs, c_glob_runs = [], [], []
    
    for s in SEEDS:
        # Φτιάχνουμε το δίκτυο HK με σταθερό m=22
        G = nx.powerlaw_cluster_graph(n=N, m=22, p=p, seed=s)
        
        edges_runs.append(G.number_of_edges())
        degs = [d for _, d in G.degree()]
        min_k_runs.append(np.min(degs))
        max_k_runs.append(np.max(degs))
        mean_k_runs.append(np.mean(degs))
        c_avg_runs.append(nx.average_clustering(G))
        c_glob_runs.append(nx.transitivity(G))
        lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        diam_runs.append(nx.diameter(lcc))
        
    results_list.append({
        "Μοντέλο": "HK",
        "Παράμετρος": p,
        "Edges_mean": np.mean(edges_runs), "Edges_std": np.std(edges_runs),
        "Min_k_mean": np.mean(min_k_runs), "Min_k_std": np.std(min_k_runs),
        "Max_k_mean": np.mean(max_k_runs), "Max_k_std": np.std(max_k_runs),
        "Mean_k_mean": np.mean(mean_k_runs), "Mean_k_std": np.std(mean_k_runs),
        "Diam_mean": np.mean(diam_runs), "Diam_std": np.std(diam_runs),
        "C_avg_mean": np.mean(c_avg_runs), "C_avg_std": np.std(c_avg_runs),
        "C_glob_mean": np.mean(c_glob_runs), "C_glob_std": np.std(c_glob_runs),
    })

# Αποθηκεύουμε τα νούμερα σε CSV για το Excel
df_results = pd.DataFrame(results_list)
df_results.to_csv("experiment_part_d_results.csv", sep=";", index=False, encoding="utf-8-sig")
print("\n-> Ο πίνακας αποθηκεύτηκε στο: experiment_part_d_results.csv")

# =============================================================================
# ΒΗΜΑ 6: ΣΧΕΔΙΑΣΗ ΔΙΑΓΡΑΜΜΑΤΩΝ ΜΕ ERROR BARS
# =============================================================================
print("-> Σχεδίαση γραφημάτων με error bars...")

models_list = ["ER", "WS", "BA", "HK"]
x_labels = [
    "Πιθανότητα σύνδεσης p (ER)",
    "Πιθανότητα rewiring p (WS - log scale)",
    "Ακμές m ανά νέο κόμβο (BA)",
    "Πιθανότητα τριγώνων p (HK)"
]

for model, xlabel in zip(models_list, x_labels):
    df_m = df_results[df_results["Μοντέλο"] == model]
    x = df_m["Παράμετρος"]

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
    # https://numpy.org/doc/stable/reference/routines.statistics.html
    # 4 υπο-γραφήματα (2x2) για κάθε μοντέλο
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Μοντέλο {model}: Μετρικές vs Παράμετρος (Mean ± Std από 5 seeds)", fontsize=13, fontweight='bold')
    
    # 1. Πλήθος Ακμών
    axes[0, 0].errorbar(x, df_m["Edges_mean"], yerr=df_m["Edges_std"], fmt='-o', color='blue', capsize=4)
    axes[0, 0].set_title("Πλήθος Ακμών")
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("Ακμές")
    axes[0, 0].grid(True)
    if model == "WS": axes[0, 0].set_xscale('log')
    
    # 2. Max, Μέσος και Min Βαθμός
    axes[0, 1].errorbar(x, df_m["Max_k_mean"], yerr=df_m["Max_k_std"], fmt='-s', color='red', label='Max k', capsize=4)
    axes[0, 1].errorbar(x, df_m["Mean_k_mean"], yerr=df_m["Mean_k_std"], fmt='-o', color='purple', label='Μέσος k', capsize=4)
    axes[0, 1].errorbar(x, df_m["Min_k_mean"], yerr=df_m["Min_k_std"], fmt='-^', color='orange', label='Min k', capsize=4)
    axes[0, 1].set_title("Στατιστικά Βαθμού")
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel("Βαθμός")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    if model == "WS": axes[0, 1].set_xscale('log')
    
    # 3. Διάμετρος στη LCC
    axes[1, 0].errorbar(x, df_m["Diam_mean"], yerr=df_m["Diam_std"], fmt='-d', color='darkgreen', capsize=4)
    axes[1, 0].set_title("Διάμετρος (LCC)")
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel("Διάμετρος")
    axes[1, 0].grid(True)
    if model == "WS": axes[1, 0].set_xscale('log')
    
    # 4. Μέσο και Global Clustering
    axes[1, 1].errorbar(x, df_m["C_avg_mean"], yerr=df_m["C_avg_std"], fmt='-o', color='teal', label='Μέσο Clustering', capsize=4)
    axes[1, 1].errorbar(x, df_m["C_glob_mean"], yerr=df_m["C_glob_std"], fmt='--^', color='olive', label='Global Clustering', capsize=4)
    axes[1, 1].set_title("Clustering Coefficient")
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel("Clustering")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    if model == "WS": axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f"part_d_experiment_{model}.png", dpi=200)
    plt.close()

print("-> Ολοκληρώθηκε με επιτυχία! Παράχθηκαν τα 4 αρχεία εικόνων και το CSV.")