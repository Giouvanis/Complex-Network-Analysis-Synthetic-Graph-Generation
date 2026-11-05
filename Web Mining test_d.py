import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΟΣ
# =============================================================================
N = 4039
# 5 διαφορετικά seeds για τις 5 ανεξάρτητες επαναλήψεις
SEEDS = [101, 202, 303, 404, 505]

# Παράμετροι ανά μοντέλο: 
# Περιλαμβάνουν την τιμή του Μέρους Α + 3 επιπλέον τιμές (σύνολο 4 τιμές ανά μοντέλο)
PARAM_SETTINGS = {
    "ER": {
        "param_name": "p (Edge Probability)",
        "param_values": [0.005, 0.01082, 0.018, 0.025],  # 0.01082 ήταν στο Μέρος Α
        "is_log_x": False
    },
    "WS": {
        "param_name": "p (Rewiring Probability)",
        "param_values": [0.001, 0.01, 0.02, 0.1],        # 0.02 ήταν στο Μέρος Α (k=44 σταθερό)
        "is_log_x": True
    },
    "BA": {
        "param_name": "m (Edges per new node)",
        "param_values": [10, 16, 22, 30],                 # 22 ήταν στο Μέρος Α
        "is_log_x": False
    },
    "HK": {
        "param_name": "p_triangle (Triangle Probability)",
        "param_values": [0.1, 0.25, 0.4, 0.7],            # 0.4 ήταν στο Μέρος Α (m=22 σταθερό)
        "is_log_x": False
    }
}

# =============================================================================
# ΣΥΝΑΡΤΗΣΗ ΠΑΡΑΓΩΓΗΣ ΔΙΚΤΥΟΥ
# =============================================================================
def generate_graph(model_type, param_val, seed):
    if model_type == "ER":
        return nx.erdos_renyi_graph(n=N, p=param_val, seed=seed)
    elif model_type == "WS":
        return nx.watts_strogatz_graph(n=N, k=44, p=param_val, seed=seed)
    elif model_type == "BA":
        return nx.barabasi_albert_graph(n=N, m=int(param_val), seed=seed)
    elif model_type == "HK":
        return nx.powerlaw_cluster_graph(n=N, m=22, p=param_val, seed=seed)
    raise ValueError(f"Άγνωστο μοντέλο: {model_type}")

# =============================================================================
# ΕΚΤΕΛΕΣΗ ΠΕΙΡΑΜΑΤΩΝ ΚΑΙ ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΤΡΙΚΩΝ
# =============================================================================
all_experiment_records = []

for model, config in PARAM_SETTINGS.items():
    print(f"\n=======================================================")
    print(f"Εκτέλεση πειράματος για το μοντέλο: {model}")
    print(f"=======================================================")
    
    for val in config["param_values"]:
        print(f"  -> Έλεγχος για {config['param_name']} = {val} ...")
        
        edges_runs = []
        min_deg_runs = []
        max_deg_runs = []
        mean_deg_runs = []
        diam_runs = []
        avg_clust_runs = []
        glob_clust_runs = []
        
        for s in SEEDS:
            # 1. Παραγωγή γράφου
            G = generate_graph(model, val, s)
            
            # 2. Ακμές & Βαθμοί
            edges_runs.append(G.number_of_edges())
            degs = [d for _, d in G.degree()]
            min_deg_runs.append(np.min(degs))
            max_deg_runs.append(np.max(degs))
            mean_deg_runs.append(np.mean(degs))
            
            # 3. Clustering
            avg_clust_runs.append(nx.average_clustering(G))
            glob_clust_runs.append(nx.transitivity(G))
            
            # 4. Διάμετρος στη μεγαλύτερη συνιστώσα (LCC)
            lcc_nodes = max(nx.connected_components(G), key=len)
            G_lcc = G.subgraph(lcc_nodes).copy()
            diam_runs.append(nx.diameter(G_lcc))
            
        # Αποθήκευση μέσης τιμής και τυπικής απόκλισης
        all_experiment_records.append({
            "Μοντέλο": model,
            "Παράμετρος": val,
            "Edges_mean": np.mean(edges_runs), "Edges_std": np.std(edges_runs),
            "Min_Deg_mean": np.mean(min_deg_runs), "Min_Deg_std": np.std(min_deg_runs),
            "Max_Deg_mean": np.mean(max_deg_runs), "Max_Deg_std": np.std(max_deg_runs),
            "Mean_Deg_mean": np.mean(mean_deg_runs), "Mean_Deg_std": np.std(mean_deg_runs),
            "Diameter_mean": np.mean(diam_runs), "Diameter_std": np.std(diam_runs),
            "Avg_Clust_mean": np.mean(avg_clust_runs), "Avg_Clust_std": np.std(avg_clust_runs),
            "Glob_Clust_mean": np.mean(glob_clust_runs), "Glob_Clust_std": np.std(glob_clust_runs),
        })

df_all = pd.DataFrame(all_experiment_records)

# Εξαγωγή του πλήρους πίνακα αποτελεσμάτων σε CSV
df_all.to_csv("experiment_d_results.csv", sep=";", index=False, encoding="utf-8-sig")
print("\n-> Όλα τα αποτελέσματα αποθηκεύτηκαν στο: experiment_d_results.csv")

# =============================================================================
# ΣΧΕΔΙΑΣΗ ΔΙΑΓΡΑΜΜΑΤΩΝ ΜΕΤΡΙΚΗ VS ΠΑΡΑΜΕΤΡΟΣ (ΜΕ ERROR BARS)
# =============================================================================
print("\n-> Δημιουργία γραφημάτων με error bars...")

for model, config in PARAM_SETTINGS.items():
    df_sub = df_all[df_all["Μοντέλο"] == model]
    x = df_sub["Παράμετρος"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Μοντέλο {model}: Επίδραση της παραμέτρου στις μετρικές (Mean ± Std)", fontsize=14, fontweight="bold")
    
    # 1. Ακμές & Μέσος Βαθμός
    axes[0, 0].errorbar(x, df_sub["Edges_mean"], yerr=df_sub["Edges_std"], fmt='-o', color='tab:blue', capsize=5, label='Ακμές')
    axes[0, 0].set_title("Πλήθος Ακμών")
    axes[0, 0].set_xlabel(config["param_name"])
    axes[0, 0].set_ylabel("Ακμές")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    if config["is_log_x"]: axes[0, 0].set_xscale('log')
    
    # 2. Min & Max Βαθμός
    axes[0, 1].errorbar(x, df_sub["Max_Deg_mean"], yerr=df_sub["Max_Deg_std"], fmt='-s', color='tab:red', capsize=5, label='Max Degree')
    axes[0, 1].errorbar(x, df_sub["Min_Deg_mean"], yerr=df_sub["Min_Deg_std"], fmt='-^', color='tab:orange', capsize=5, label='Min Degree')
    axes[0, 1].set_title("Ελάχιστος & Μέγιστος Βαθμός")
    axes[0, 1].set_xlabel(config["param_name"])
    axes[0, 1].set_ylabel("Βαθμός (k)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    if config["is_log_x"]: axes[0, 1].set_xscale('log')
    
    # 3. Διάμετρος
    axes[1, 0].errorbar(x, df_sub["Diameter_mean"], yerr=df_sub["Diameter_std"], fmt='-d', color='tab:purple', capsize=5)
    axes[1, 0].set_title("Διάμετρος (LCC)")
    axes[1, 0].set_xlabel(config["param_name"])
    axes[1, 0].set_ylabel("Διάμετρος")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    if config["is_log_x"]: axes[1, 0].set_xscale('log')
    
    # 4. Clustering Coefficients (Average & Global)
    axes[1, 1].errorbar(x, df_sub["Avg_Clust_mean"], yerr=df_sub["Avg_Clust_std"], fmt='-o', color='tab:green', capsize=5, label='Average Clustering')
    axes[1, 1].errorbar(x, df_sub["Glob_Clust_mean"], yerr=df_sub["Glob_Clust_std"], fmt='--^', color='tab:olive', capsize=5, label='Global Clustering')
    axes[1, 1].set_title("Clustering Coefficient")
    axes[1, 1].set_xlabel(config["param_name"])
    axes[1, 1].set_ylabel("Clustering")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    if config["is_log_x"]: axes[1, 1].set_xscale('log')
    
    plt.tight_layout()
    plot_filename = f"D_experiment_{model}.png"
    plt.savefig(plot_filename, dpi=200)
    plt.close()
    print(f"  -> Το γράφημα για το {model} αποθηκεύτηκε στο: {plot_filename}")

print("\n-> Το πείραμα του Μέρους Δ ολοκληρώθηκε επιτυχώς!")