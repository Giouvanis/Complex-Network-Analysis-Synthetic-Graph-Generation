import os
import gzip
import shutil
import urllib.request
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# ΒΗΜΑ 1: ΦΟΡΤΩΣΗ ΤΟΥ ΠΡΑΓΜΑΤΙΚΟΥ ΔΙΚΤΥΟΥ FACEBOOK (i)
# =============================================================================
# Το αρχείο που ζητάει η εκφώνηση βρίσκεται στη σελίδα του SNAP.
# Εδώ ελέγχουμε αν υπάρχει ήδη στον φάκελο. Αν όχι, το κατεβάζει μόνο του.
filename = "facebook_combined.txt"
gz_filename = "facebook_combined.txt.gz"
url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"

if not os.path.exists(filename):
    print("-> Κατέβασμα του αρχείου Facebook από το SNAP...")
    urllib.request.urlretrieve(url, gz_filename)
    # Αποσυμπίεση του αρχείου .gz σε κανονικό αρχείο κειμένου .txt
    with gzip.open(gz_filename, "rb") as f_in:
        with open(filename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("-> Το αρχείο αποσυμπιέστηκε επιτυχώς!")

# Διαβάζουμε το δίκτυο. Είναι μη κατευθυνόμενο (Graph)
G_fb = nx.read_edgelist(filename, nodetype=int, create_using=nx.Graph())
print("-> Το δίκτυο Facebook φορτώθηκε με επιτυχία.")

# =============================================================================
# ΒΗΜΑ 2: ΥΠΟΛΟΓΙΣΜΟΣ ΠΑΡΑΜΕΤΡΩΝ ΓΙΑ ΤΑ ΣΥΝΘΕΤΙΚΑ ΔΙΚΤΥΑ
# =============================================================================
# Η εκφώνηση ζητάει τα συνθετικά δίκτυα να έχουν 4039 κόμβους και αντίστοιχες
# ιδιότητες (ίδιο μέσο βαθμό) με το πραγματικό.
N = G_fb.number_of_nodes()       # Πλήθος κόμβων: 4039
M_fb = G_fb.number_of_edges()    # Πλήθος ακμών: 88234

# Ο μέσος βαθμός είναι: <k> = (2 * Ακμές) / Κόμβοι
avg_k = (2 * M_fb) / N           # Περίπου 43.69
print(f"Facebook: Κόμβοι = {N}, Ακμές = {M_fb}, Μέσος Βαθμός = {avg_k:.2f}")

# 1. Παράμετρος για Erdős-Rényi (ER):
# Στο ER η πιθανότητα σύνδεσης p επιλέγεται ώστε ο μέσος βαθμός να είναι <k>:
# p = <k> / (N - 1)
p_er = avg_k / (N - 1)

# 2. Παράμετρος για Watts-Strogatz (WS):
# Το k_ws πρέπει να είναι άρτιος ακέραιος (πόσοι κοντινοί γείτονες συνδέονται αρχικά).
# Το 44 είναι ο πλησιέστερος άρτιος στο 43.69.
k_ws = 44
# Το p_ws είναι η πιθανότητα επανασύνδεσης ακμής. Επιλέγουμε 0.02 για να έχουμε
# το φαινόμενο Small-World (υψηλό clustering, μικρές αποστάσεις).
p_ws = 0.02

# 3. Παράμετρος για Barabási-Albert (BA):
# Κάθε νέος κόμβος φέρνει m ακμές. Ο τελικός μέσος βαθμός είναι περίπου 2*m.
# Επομένως: m = round(avg_k / 2) = round(43.69 / 2) = 22.
m_ba = 22

# 4. Παράμετρος για Holme-Kim (HK - Μέρος Γ):
# Χρησιμοποιεί το ίδιο m=22 όπως το BA και μία πιθανότητα p_hk για κλείσιμο τριγώνων.
p_hk = 0.4

# Ορίζουμε σταθερό seed για αναπαραγωγιμότητα
random_seed = 42

print("\n-> Δημιουργία συνθετικών δικτύων...")
# (ii) Erdős-Rényi
G_er = nx.erdos_renyi_graph(n=N, p=p_er, seed=random_seed)
# (iii) Watts-Strogatz
G_ws = nx.watts_strogatz_graph(n=N, k=k_ws, p=p_ws, seed=random_seed)
# (iv) Barabási-Albert
G_ba = nx.barabasi_albert_graph(n=N, m=m_ba, seed=random_seed)
# (v) Holme-Kim (Μέρος Γ)
G_hk = nx.powerlaw_cluster_graph(n=N, m=m_ba, p=p_hk, seed=random_seed)

# Βάζουμε όλα τα δίκτυα σε ένα λεξικό για να τα διατρέχουμε εύκολα
all_graphs = {
    "Facebook": G_fb,
    "Erdos-Renyi (ER)": G_er,
    "Watts-Strogatz (WS)": G_ws,
    "Barabasi-Albert (BA)": G_ba,
    "Holme-Kim (HK)": G_hk
}

# =============================================================================
# ΒΗΜΑ 3: ΥΠΟΛΟΓΙΣΜΟΣ ΜΕΤΡΗΣΕΩΝ (ΜΕΡΟΣ Α: α έως θ)
# =============================================================================
table_data = []

for name, G in all_graphs.items():
    print(f"\nΥπολογισμός μετρικών για: {name} ... (παρακαλώ περιμένετε)")

    # https://networkx.org/documentation/stable/reference/classes/graph.html
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.number_of_nodes.html
    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.number_of_edges.html
    # α) Πλήθος κόμβων και ακμών
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.selfloop.number_of_selfloops.html
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isolate.number_of_isolates.html
    # β) Κόμβοι με self-loop και απομονωμένοι κόμβοι
    self_loops = nx.number_of_selfloops(G)
    isolated_nodes = nx.number_of_isolates(G)

    # https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.degree.html
    # https://numpy.org/doc/stable/reference/routines.statistics.html
    # γ) Στατιστικά βαθμού
    degrees = [d for node, d in G.degree()]
    max_deg = np.max(degrees)
    min_deg = np.min(degrees)
    mean_deg = np.mean(degrees)
    median_deg = np.median(degrees)
    var_deg = np.var(degrees)  # Μέσος όρος των τετραγώνων των αποκλίσεων από τον μέσο

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
    # ζ) Συνεκτικές συνιστώσες και Μεγαλύτερη Συνεκτική Συνιστώσα (LCC)
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_cc_nodes = max(components, key=len)
    G_lcc = G.subgraph(largest_cc_nodes).copy()
    lcc_nodes = G_lcc.number_of_nodes()
    lcc_edges = G_lcc.number_of_edges()

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.diameter.html
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.distance_measures.radius.html
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.average_shortest_path_length.html
    # δ & η) Διάμετρος, Ακτίνα και Μέσο Μονοπάτι στη LCC
    # Υπολογίζουμε τις αποστάσεις όλων των ζευγών ώστε να εκτελεστεί γρήγορα
    print(f"   -> Υπολογισμός αποστάσεων στη LCC για το {name}...")
    lengths_iterator = nx.all_pairs_shortest_path_length(G_lcc)
    eccentricities = {}
    sum_path_lengths = 0
    total_pairs = lcc_nodes * (lcc_nodes - 1)
    
    for src, targets in lengths_iterator:
        eccentricities[src] = max(targets.values())
        sum_path_lengths += sum(targets.values())
        
    diameter = max(eccentricities.values())
    radius = min(eccentricities.values())
    avg_path_length = sum_path_lengths / total_pairs

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html
    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html
    # ε) Clustering coefficient
    avg_clustering = nx.average_clustering(G)
    global_clustering = nx.transitivity(G)

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.assortativity.degree_assortativity_coefficient.html
    # στ) Assortativity coefficient
    assortativity = nx.degree_assortativity_coefficient(G)

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.triangles.html
    # θ) Πλήθος τριγώνων
    # Η nx.triangles(G) επιστρέφει λεξικό με τα τρίγωνα ανά κόμβο.
    # Κάθε τρίγωνο μετριέται 3 φορές, οπότε διαιρούμε με 3.
    num_triangles = sum(nx.triangles(G).values()) // 3
    
    # Αποθήκευση στη λίστα των αποτελεσμάτων
    table_data.append({
        "Δίκτυο": name,
        "Κόμβοι": num_nodes,
        "Ακμές": num_edges,
        "Self-loops": self_loops,
        "Απομονωμένοι": isolated_nodes,
        "Min Βαθμός": min_deg,
        "Max Βαθμός": max_deg,
        "Μέσος Βαθμός": round(mean_deg, 2),
        "Διάμεσος Βαθμός": median_deg,
        "Διασπορά Βαθμού": round(var_deg, 2),
        "Κόμβοι LCC": lcc_nodes,
        "Ακμές LCC": lcc_edges,
        "Συνιστώσες": num_components,
        "Διάμετρος (LCC)": diameter,
        "Ακτίνα (LCC)": radius,
        "Μέσο Μονοπάτι (LCC)": round(avg_path_length, 3),
        "Μέσο Clustering": round(avg_clustering, 4),
        "Global Clustering": round(global_clustering, 4),
        "Assortativity": round(assortativity, 4),
        "Τρίγωνα": num_triangles
    })

# Μετατροπή σε DataFrame και εκτύπωση του πίνακα
df_metrics = pd.DataFrame(table_data)
print("\n" + "="*80)
print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΜΕΤΡΗΣΕΩΝ (ΜΕΡΟΣ Α)")
print("="*80)
print(df_metrics.to_string(index=False))

# Αποθήκευση σε αρχείο Excel/CSV
df_metrics.to_csv("results_table.csv", index=False, encoding="utf-8-sig")
print("\n-> Ο πίνακας αποθηκεύτηκε στο αρχείο: results_table.csv")

# =============================================================================
# ΒΗΜΑ 4: ΣΧΕΔΙΑΣΗ ΔΙΑΓΡΑΜΜΑΤΩΝ ΓΙΑ ΚΑΘΕ ΔΙΚΤΥΟ (α έως στ)
# =============================================================================

print("\n-> Έναρξη παραγωγής διαγραμμάτων...")

for name, G in all_graphs.items():
    print(f"   -> Σχεδίαση για: {name}")
    degrees = np.array([d for node, d in G.degree()])
    
    # Δημιουργούμε παράθυρο 2 γραμμών x 3 στηλών
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Διαγράμματα Ανάλυσης Βαθμού & Clustering: {name}", fontsize=14, fontweight='bold')

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    # α) Ιστόγραμμα βαθμού σε γραμμική κλίμακα
    axes[0, 0].hist(degrees, bins=30, color="skyblue", edgecolor="black", density=True)
    axes[0, 0].set_title("α) Linear Degree Histogram")
    axes[0, 0].set_xlabel("Βαθμός (k)")
    axes[0, 0].set_ylabel("Συχνότητα P(k)")

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html
    # β) Ιστόγραμμα βαθμού σε log-log κλίμακα
    deg_vals, counts = np.unique(degrees, return_counts=True)
    prob_vals = counts / len(degrees)
    axes[0, 1].scatter(deg_vals, prob_vals, color="navy", alpha=0.7, s=20)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("β) Log-Log Degree Scatter")
    axes[0, 1].set_xlabel("Βαθμός (k)")
    axes[0, 1].set_ylabel("P(k)")

    # https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    # γ) Ιστόγραμμα με log binning σε log-log κλίμακα
    # Δημιουργούμε κάδους με εκθετικά αυξανόμενο μέγεθος
    min_k = max(1, degrees.min())
    max_k = degrees.max()
    log_bins = np.logspace(np.log10(min_k), np.log10(max_k), num=15)
    hist_vals, edges = np.histogram(degrees, bins=log_bins, density=True)
    # Παίρνουμε το γεωμετρικό κέντρο κάθε κάδου
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    # Κρατάμε μόνο όσους κάδους έχουν τιμές > 0
    valid = hist_vals > 0
    axes[0, 2].plot(bin_centers[valid], hist_vals[valid], "o-", color="firebrick")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("γ) Log-Binning (Log-Log)")
    axes[0, 2].set_xlabel("Βαθμός (k)")
    axes[0, 2].set_ylabel("P(k)")

    # https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html
    # δ) Συσσωρευτική κατανομή βαθμού (CCDF) σε log-log κλίμακα
    # P(X >= k): Πιθανότητα ένας κόμβος να έχει βαθμό μεγαλύτερο ή ίσο του k
    sorted_deg = np.sort(degrees)
    ccdf = 1.0 - (np.arange(len(sorted_deg)) / float(len(sorted_deg)))
    axes[1, 0].plot(sorted_deg, ccdf, color="purple", linewidth=2)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("δ) Cumulative Degree (CCDF)")
    axes[1, 0].set_xlabel("Βαθμός (k)")
    axes[1, 0].set_ylabel("P(X >= k)")

    # https://networkx.org/documentation/networkx-2.8.4/auto_examples/graph/plot_degree_sequence.html
    # ε) Zipf κατανομή βαθμού σε log-log κλίμακα
    # Κατάταξη των βαθμών από τον μεγαλύτερο στον μικρότερο (Rank vs Degree)
    ranks = np.arange(1, len(sorted_deg) + 1)
    axes[1, 1].plot(ranks, sorted_deg[::-1], color="darkgreen", linewidth=2)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("ε) Zipf Plot (Rank vs Degree)")
    axes[1, 1].set_xlabel("Κατάταξη (Rank)")
    axes[1, 1].set_ylabel("Βαθμός (k)")

    # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html
    # στ) Διάγραμμα διασποράς βαθμού και clustering με μέση καμπύλη C(k)
    node_clustering = nx.clustering(G)
    # Υπολογισμός του μέσου όρου clustering για κάθε τιμή βαθμού k
    k_to_c = {}
    for node, k in G.degree():
        c_val = node_clustering[node]
        if k not in k_to_c:
            k_to_c[k] = []
        k_to_c[k].append(c_val)
        
    unique_k = np.array(sorted(k_to_c.keys()))
    mean_c_per_k = np.array([np.mean(k_to_c[k]) for k in unique_k])
    
    all_k = [G.degree(n) for n in G.nodes()]
    all_c = [node_clustering[n] for n in G.nodes()]
    axes[1, 2].scatter(all_k, all_c, alpha=0.1, color="gray", s=10, label="Κόμβοι")
    axes[1, 2].plot(unique_k, mean_c_per_k, color="red", linewidth=2, label="Μέσο C(k)")
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_title("στ) Degree vs Clustering C(k)")
    axes[1, 2].set_xlabel("Βαθμός (k)")
    axes[1, 2].set_ylabel("Clustering Coefficient C")
    axes[1, 2].legend()
    
    plt.tight_layout()
    plot_filename = f"plot_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(plot_filename, dpi=200)
    plt.close()

print("-> Όλα τα διαγράμματα αποθηκεύτηκαν επιτυχώς σε εικόνες PNG!")