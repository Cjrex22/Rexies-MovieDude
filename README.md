<div align="center">

# 🎬 MovieDude
### **Intelligent Latent Movie Recommendation System**
*Powered by Dual K-Means Clustering, Latent 2D PCA Cartography & Content-Collaborative Fusion*

<br>

[![Python](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Holdout Hit Rate](https://img.shields.io/badge/Hit_Rate@5-84.00%25-2ea44f?style=for-the-badge)]()
[![Inference Latency](https://img.shields.io/badge/Latency-%3C_45ms-blueviolet?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br>

[**🚀 Quick Start**](#-quick-start) •
[**✨ Key Features**](#-key-features) •
[**🧠 System Architecture**](#-system-architecture) •
[**📊 Benchmarks & Evaluation**](#-benchmarks--evaluation) •
[**🗄️ Database Design**](#-database-design) •
[**💻 CLI Tour**](#-interactive-cli-tour)

---

</div>

## 📌 Overview

**MovieDude** is a machine learning-driven movie recommendation engine designed to bridge **structured relational database storage** with **unsupervised geometric learning**. 

Traditional recommendation engines often fall into one of two traps: naive genre filtering (which ignores tone, stylistic nuances, and artistic depth) or pure collaborative filtering (which suffers from acute cold-start penalties and high latency).

MovieDude resolves this by partitioning both the **6,000+ movie manifold** ($K=10$ item clusters) and the **600+ user interaction manifold** ($K=6$ peer cohorts) using unsupervised **K-Means Clustering**. Candidate retrieval is bounded by latent taste clusters for sub-50ms latency, while final rankings are determined by **High-Dimensional Cosine Similarity** blended with **Peer Cohort Consensus**.

---

## 🚀 Quick Start

Get MovieDude running in **3 simple commands**:

### 1. Open Terminal in the project directory
```bash
cd movieml
```

### 2. Install dependencies (Run once)
```bash
pip install -r requirements.txt
```
> [!NOTE]
> If you prefer an isolated virtual environment:
> - **macOS / Linux:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
> - **Windows:** `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`

### 3. Launch the application
```bash
python3 src/main.py
```
*(On Windows, run `python src/main.py`)*

---

### 🔑 Pre-Configured Test Accounts

The database includes two clean user profiles ready for testing:

| User ID | Password | Profile Status |
|:---:|:---:|---|
| **`Aryan`** | `12345` | Ready to log ratings and generate personalized feeds |
| **`Churchil`** | `12345` | Ready to log ratings and generate personalized feeds |

---

## ✨ Key Features

- **Dual K-Means Clustering**: Unsupervised geometric partitioning of 6,069 titles into $K=10$ semantic film clusters and 612 users into $K=6$ behavioral taste cohorts.
- **Sub-50ms Candidate Pruning**: Latent cluster candidate filtering avoids exhaustive $O(N \cdot D)$ scans, delivering near-instant recommendations.
- **Interactive 2D PCA Cartography**: Generates an interactive HTML5 latent map (`reports/movie_clusters_pca.html`) allowing users to visually inspect movie clusters directly in their browser.
- **★ 7.5 Quality Gate**: Lukewarm ratings ($< 7.5/10$) are logged in history but excluded from the taste centroid so they don't skew recommendations.
- **Dynamic Rating-Power Weighting**: $w_i = \max(0.5, r_i - 6.5)$ ensures stronger favorites carry higher mathematical influence (e.g., an 8.5 rating has double the weight of a 7.5).
- **Franchise Deduplication**: Regex and normalized Levenshtein clustering collapses sequels and spin-offs so a single franchise cannot dominate recommendation slots.
- **Typo & Accent-Proof Search**: Unicode diacritic stripping (`Bāhubali` $\rightarrow$ `Bahubali`) and fuzzy sequence matching with interactive disambiguation prompts for multi-part releases.
- **Instant In-Memory Updates**: Adding a rating triggers atomic cache invalidation, updating recommendations in real time without restarting.

---

## 🧠 System Architecture

```
   ┌────────────────────────────────────────────────────────┐
   │             Raw Movie Metadata & User Logs             │
   │           (6,069 Movies • 83,911 User Ratings)         │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │               Feature Engineering Layer                │
   │  • 19 Multi-Hot Genres with L2 Footprint Normalization │
   │  • TF-IDF Thematic Keywords & Plot Motifs              │
   └─────────────┬────────────────────────────┬─────────────┘
                 │                            │
                 ▼                            ▼
   ┌───────────────────────────┐┌───────────────────────────┐
   │  Item Clustering (K = 10) ││  User Clustering (K = 6) │
   │  • Lloyd's + K-Means++    ││  • Behavioral Cohorts    │
   │  • Inertia: 4,194.06      ││  • Silhouette: +0.1358   │
   │  • Silhouette: +0.0636    ││  • Peer Consensus Groups │
   └─────────────┬─────────────┘└─────────────┬─────────────┘
                 │                            │
                 ├────────────────────────────┤
                 ▼                            ▼
   ┌───────────────────────────┐┌───────────────────────────┐
   │  2D PCA Visualization     ││  Hybrid Recommendation    │
   │  • Eigen-Decomposition    ││  • Quality Gate: r >= 7.5 │
   │  • Interactive HTML5 Map  ││  • S_final = 0.5*Content  │
   │  • Spatial Orientation    ││           + 0.5*Peer      │
   │  (Inspection Only)        ││  • Franchise Deduplication│
   └───────────────────────────┘└───────────────────────────┘
```

### 1. High-Dimensional Feature Representation
- **Multi-Hot Categorical Encoding**: 19 distinct film genres converted into indicator vectors.
- **Per-Item L2 Footprint Normalization**:
  $$\mathbf{g}_i^{\text{norm}} = \frac{\mathbf{g}_i}{\|\mathbf{g}_i\|_2}$$
  Ensures that multi-genre movies (e.g., Action+Sci-Fi+Thriller+Crime) do not artificially overpower focused, single-genre artistic cinema.
- **TF-IDF Keyword Weighting**: Penalizes common filler words while giving high mathematical importance to distinct thematic motifs (e.g., *police interrogation*, *alibi*, *stockbroker*).

### 2. Dual K-Means Clustering
- **Item Space ($K=10$)**: Discovers 10 natural movie categories:
  - *Cluster #0:* Thriller & Action
  - *Cluster #1:* Romance & Drama
  - *Cluster #2:* Crime & Neo-Noir
  - *Cluster #3:* Action & Adventure
  - *Cluster #4:* Comedy & Friendship
  - *Cluster #5:* Sci-Fi & Dystopia
  - *Cluster #6:* Animation & Family
  - *Cluster #7:* Drama & Biography
  - *Cluster #8:* Horror & Suspense
  - *Cluster #9:* Historical Drama
- **User Space ($K=6$)**: Discovers 6 distinct peer taste communities from 83,000+ interaction logs (Silhouette Score: `+0.1358`).

### 3. Dimensionality Reduction (2D PCA)
- Compresses multi-dimensional manifolds into 2D orthogonal coordinates using **Principal Component Analysis**:
  - *Movie Item PCA Variance:* $\text{PC1} = 6.39\%, \text{PC2} = 5.38\%$ (Total: $11.77\%$).
  - *User Cohort PCA Variance:* $\text{PC1} = 10.37\%, \text{PC2} = 8.84\%$ (Total: $19.21\%$).
- > [!IMPORTANT]
  > **2D PCA is used strictly for human visual orientation** in the generated browser report. Recommendation rankings are calculated on full high-dimensional vectors to avoid information compression loss.

### 4. Hybrid Scoring Formula
$$S_{\text{final}}(u, m) = 0.5 \cdot \text{Sim}_{\text{content}}(\mathbf{u}, \mathbf{x}_m) + 0.5 \cdot \text{Support}_{\text{peer}}(c_u, m)$$

- **Content Similarity**: High-dimensional Cosine Similarity between candidate vector $\mathbf{x}_m$ and user taste centroid $\mathbf{u}$:
  $$\text{CosineSim}(\mathbf{u}, \mathbf{x}_m) = \frac{\mathbf{u} \cdot \mathbf{x}_m}{\|\mathbf{u}\|_2 \|\mathbf{x}_m\|_2}$$
- **Peer Cohort Support**: Normalized rating density from users inside the same K-Means taste cohort $c_u$.
- **Strict Monotonic Ordering**: Recommendations are sorted in strictly descending order with transparent score breakdowns.

---

## 📊 Benchmarks & Evaluation

Evaluated on an **offline 80/20 train/test holdout benchmark across 100 real users** with zero data leakage:

| Evaluation Metric | Top-5 ($K=5$) | Top-10 ($K=10$) | Plain-English Meaning |
|---|:---:|:---:|---|
| **Hit Rate@K** | **84.00%** | **84.00%** | **84 out of 100 users** found a movie they loved in their top recommendations |
| **Precision@K** | **49.60%** | **24.80%** | Nearly **1 in 2 recommended movies** matched actual holdout preferences |
| **Recall@K** | **12.82%** | **12.82%** | Proportion of total test holdout items successfully retrieved |
| **Query Latency** | **< 45 ms** | **< 48 ms** | Sub-50 millisecond inference time per user query |

---

## 🗄️ Database Design

The storage layer is an embedded, ACID-compliant **SQLite3** relational database (`database/MovieDude.db`):

```text
MovieDude.db
├── Movies_sorted  (6,069 released films, metadata, keywords, ratings)
├── Users_data     (83,911 user ratings, user_rate, liked boolean flag)
└── Users          (User credentials, authentication, timestamps)
```

- **Composite Primary Keys**: `PRIMARY KEY (user_id, movie_id)` guarantees data uniqueness.
- **Atomic Upserts**: `INSERT ... ON CONFLICT DO UPDATE` allows immediate rating updates without table locks.
- **Cache Management**: Database write events atomically evict the in-memory candidate cache, guaranteeing zero staleness on immediate profile refresh.

---

## 💻 Interactive CLI Tour

The terminal application features an interactive 7-option menu:

```text
--------------------------------------------------
1 : Discover Movies Based on Your Activities (User Taste Clustering)
2 : Find Similar Movies by Given Movie Title (Item Clustering & Latent Space)
3 : Rate or Log a Movie
4 : View My Watch & Rating History
5 : Explore Movie Clusters & PCA Visualizer (Interactive Latent Map)
6 : Switch User / Log Out
7 : Exit
--------------------------------------------------
```

- **`1` Discover Movies**: Generates personalized Top 10 recommendations with transparent score breakdowns (`Final Rank Score`, `Content Sim`, `Peer Cohort Support`). For new accounts, provides a diverse cold-start starter pack across all 10 clusters.
- **`2` Find Similar Movies**: Input any movie (e.g. *Inception*, *La La Land*) to find similar films with franchise deduplication.
- **`3` Rate or Log a Movie**: Search and rate any title from 1.0 to 10.0 stars with instant profile persistence.
- **`4` View History**: Inspect all movies you have rated with color-coded seed tags (`[Seed: Rated ≥ 7.5]` vs `[Excluded: Rated < 7.5]`).
- **`5` PCA Visualizer**: Inspect live clustering metrics and auto-launch the interactive 2D cluster map in your default web browser.
- **`6` Switch User**: Log out and switch between profiles (`Aryan`, `Churchil`).
- **`7` Exit**: Safely close the application.

---

## 📁 Project Structure

```text
movieml/
├── database/
│   └── MovieDude.db                     # SQLite database (6,069 films, 83k ratings)
├── reports/
│   ├── movie_clusters_pca.html          # Interactive 2D PCA browser visualizer
│   └── movie_clusters_pca.svg           # Vector PCA cluster graphic
├── src/
│   ├── main.py                          # CLI application entry point
│   ├── clustering_benchmark.py          # Machine learning benchmark script
│   ├── evaluation.py                    # Holdout cross-validation module
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── clustering.py                # K-Means clustering, PCA & franchise deduplication
│   │   ├── engine.py                    # Hybrid recommendation engine
│   │   ├── find_by_title.py             # Diacritic-safe search & version disambiguation
│   │   ├── find_user_interests.py       # Taste profiling & 7.5 rating threshold gating
│   │   └── watched_movies.py            # User history retrieval
│   └── scripts/
│       └── login.py                     # User authentication
├── requirements.txt                     # Dependency specifications
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
