# Complex-Network-Analysis-Synthetic-Graph-Generation
This repository provides an empirical and theoretical analysis of complex networks using NetworkX in Python. It investigates the structural properties of a real-world social network (Stanford SNAP Facebook Combined Dataset) and compares it against four canonical synthetic graph generation models (ER, WS, BA, and HK).
# 📌 Project Overview

The project is divided into two primary analytical phases:

### 1. Comparative Baseline Analysis (Parts A & B)
* **Real Network Benchmark**: Automated retrieval, extraction, and parsing of the SNAP Facebook dataset ($N = 4039$ nodes, $M = 88234$ edges, average degree $\langle k \rangle \approx 43.69$).
* **Synthetic Graph Construction**: Calibration of 4 synthetic network models to mirror the empirical scale ($N = 4039$) and mean connectivity:
  * **Erdős-Rényi (ER)**: Random graph matching edge density with $p = \frac{\langle k \rangle}{N - 1}$.
  * **Watts-Strogatz (WS)**: Small-world ring lattice with fixed $k = 44$ and rewiring probability $p = 0.02$.
  * **Barabási-Albert (BA)**: Scale-free preferential attachment network with $m = 22$ edges added per incoming node.
  * **Holme-Kim (HK)**: Scale-free network tuned for high clustering using $m = 22$ and triad closure probability $p = 0.4$.
* **Topological Metric Extraction**: Calculation of degree distributions, degree variance, Connected Components, Largest Connected Component (LCC), diameter, radius, average shortest path length, average and global clustering coefficients, assortativity, and closed triangles.
* **6-Panel Distribution Visualization**: Automated plotting suite for each network featuring linear degree histogram, log-log degree scatter, logarithmic binning, cumulative degree distribution (CCDF), Zipf rank-frequency plot, and degree vs. local clustering $k$ vs. $C(k)$.

### 2. Multi-Seed Parameter Sensitivity Analysis (Part D)
* **Parametric Sweep**: Systematic exploration of how shifting key control parameters ($p$ and $m$) impacts network topology across all four synthetic models.
* **Monte Carlo Simulations**: 5 independent runs per parameter value using distinct pseudo-random seeds (`101, 202, 303, 404, 505`).
* **Statistical Bounds**: Calculation of mean and standard deviation ($\text{Mean} \pm \text{Std}$) across edge counts, minimum/maximum/average degrees, LCC diameter, and clustering coefficients.
* **Error-Bar Visualizations**: $2 \times 2$ grid plots per model with standard deviation error bars illustrating metric responsiveness to parameter scaling.

## 🛠 Tech Stack & Dependencies

* **Language**: Python 3.9+
* **Graph Modeling**: [NetworkX](https://networkx.org/)
* **Numerical Computing**: [NumPy](https://numpy.org/)
* **Tabular Data Processing**: [Pandas](https://pandas.pydata.org/)
* **Data Visualization**: [Matplotlib](https://matplotlib.org/)

## 📂 Repository Structure

* `Network Analysis Project_1(AB).py`: Script that fetches the SNAP dataset, constructs all 5 baseline networks, extracts topological metrics, and renders the 6-panel distribution figures.
* `Web Mining test_d.py` / `Network Analysis Project_1(D).py`: Script that executes the multi-seed parameter sweeps, runs statistical aggregations, and generates the $2 \times 2$ error-bar evaluation charts.
* `results_table.csv`: Structured tabular export of all topological metrics for the baseline networks.
* `experiment_d_results.csv`: Statistical records ($\text{Mean} \pm \text{Std}$) resulting from the parameter sweep experiments.
* `*.png`: Output plots displaying degree distributions, clustering spectra, and parameter sensitivity curves.

## 📊 Summary of Baseline Results

| Model | Nodes ($N$) | Edges ($M$) | Mean Degree $\langle k \rangle$ | Diameter (LCC) | Average Clustering ($C$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Facebook (SNAP)** | 4,039 | 88,234 | ~43.69 | 8 | 0.6055 |
| **Erdős-Rényi (ER)** | 4,039 | ~88,200 | ~43.69 | 3 | ~0.0108 |
| **Watts-Strogatz (WS)** | 4,039 | 88,858 | 44.00 | 5 | ~0.6500 |
| **Barabási-Albert (BA)** | 4,039 | 88,374 | ~43.76 | 4 | ~0.0330 |
| **Holme-Kim (HK)** | 4,039 | 88,374 | ~43.76 | 4 | ~0.1500 |

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install networkx numpy pandas matplotlib
