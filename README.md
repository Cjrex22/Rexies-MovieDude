<div align="center">

# 🎬 MovieDude
### Intelligent Latent Movie Recommendation Engine
**Powered by Dual K-Means Clustering, 2D PCA Cartography & Content-Collaborative Fusion**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Accuracy](https://img.shields.io/badge/Hit_Rate@5-84.0%25-brightgreen?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Key Features</a> •
  <a href="#-machine-learning-architecture">ML Architecture</a> •
  <a href="#-benchmarks--evaluation">Benchmarks</a> •
  <a href="#-database-design">Database Design</a> •
  <a href="#-cli-tour">CLI Tour</a>
</p>

</div>

---

## 📌 Overview

**MovieDude** is a machine learning-driven movie recommendation platform combining an embedded **ACID-compliant SQLite relational database** with **Unsupervised Machine Learning**. 

Rather than relying on naive genre matching or slow, un-indexed scans, MovieDude partitions both the 6,000+ film catalog and the 600+ user interaction space into latent clusters. Recommendations are ranked using **high-dimensional Cosine Similarity** fused with **Peer Cohort Consensus**, with interactive **2D PCA projection** for spatial visualization.

---

## 🚀 Quick Start

Get MovieDude running in **3 simple steps**:

### 1. Open Terminal in the project folder
```bash
cd movieml
```

### 2. Install dependencies (Run once)
```bash
pip install -r requirements.txt
```
> [!NOTE]
> If you prefer a virtual environment:
> - **macOS / Linux:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
> - **Windows:** `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`

### 3. Start the application
```bash
python3 src/main.py
```
*(On Windows, use `python src/main.py`)*

---

### 🔑 Pre-Configured Login Accounts

When prompted, log in with either of the following accounts:

| User ID | Password | Purpose / Status |
|:---:|:---:|---|
| **`Aryan`** | `12345` | Clean profile ready to log movies & build a custom recommendation feed |
| **`Churchil`** | `12345` | Clean profile ready to log movies & build a custom recommendation feed |

---

## ✨ Key Features

- **Dual K-Means Clustering**: Partitions 6,000+ movies into $K=10$ latent item clusters and 600+ users into $K=6$ behavioral taste cohorts.
- **Sub-50ms Candidate Pruning**: Filters search space to the user's active clusters, delivering instant recommendations.
- **2D PCA Visualization**: Projects multi-dimensional movie and user topologies into an interactive HTML5 cartography map (`reports/movie_clusters_pca.html`).
- **★ 7.5 Quality Gating**: Lukewarm or disliked movies are recorded in history but excluded from the taste centroid so they don't dilute recommendations.
- **Franchise Deduplication**: Collapses sequels and spin-offs (e.g., *Toy Story 1, 2, 3*) so only the top-scoring entry occupies a recommendation slot.
- **Typo & Accent-Proof Search**: Unicode diacritic stripping (`Bāhubali` $\rightarrow$ `Bahubali`) with fuzzy matching and multi-part disambiguation (*Drishyam*, *K.G.F*).
- **Instant In-Memory Updates**: Adding a rating immediately invalidates cache and refreshes personalized recommendations with zero server downtime.

---

## 🧠 Machine Learning Architecture

```
   ┌────────────────────────────────────────────────────────┐
   │         User Ratings & Watch History (SQLite)          │
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
   │  • K-Means++ Centroids    ││  • Behavioral Cohorts    │
   │  • Inertia: 4,194.06      ││  • Silhouette: +0.1358   │
   │  • Silhouette: +0.0636    ││  • Peer Taste Groups     │
   └─────────────┬─────────────┘└─────────────┬─────────────┘
                 │                            │
                 ├────────────────────────────┤
                 ▼                            ▼
   ┌───────────────────────────┐┌───────────────────────────┐
   │  2D PCA Visualization     ││  Hybrid Recommendation    │
   │  • Eigen-Decomposition    ││  • Gate: r >= 7.5 only    │
   │  • Interactive HTML5 Map  ││  • Sim = 0.5*Content +    │
   │  • Spatial Orientation    ││          0.5*Peer_Support │
   │  (Inspection Only)        ││  • Franchise Deduplication│
   └───────────────────────────┘└───────────────────────────┘
```

### 1. High-Dimensional Vectorization & Footprint Normalization
- **Multi-Hot Genre Vectors**: 19 distinct genres mapped to numerical coordinates.
- **L2 Footprint Normalization**: Multi-genre titles (e.g., Action+Sci-Fi+Thriller+Crime) are normalized so they do not artificially overpower single-genre art films.
- **TF-IDF Keyword Weighting**: High weights are assigned to distinctive story elements (*alibi*, *interrogation*, *stockbroker*) while filtering out ubiquitous filler words.

### 2. Dual K-Means Clustering
- **Item Space ($K=10$)**: Partitions movies into 10 semantic categories (Crime/Neo-Noir, Romance/Drama, Sci-Fi/Dystopia, Animation/Family, Biographies/History).
- **User Space ($K=6$)**: Discovers 6 behavioral communities from 83,000+ interaction patterns (Silhouette: `+0.1358`).

### 3. Dimensionality Reduction (2D PCA)
- Compresses multi-dimensional item manifolds into 2D orthogonal coordinates via **Principal Component Analysis**.
- > [!IMPORTANT]
  > **2D PCA is used strictly for human visualization** (`reports/movie_clusters_pca.html`). Recommendation rankings are calculated on full high-dimensional feature vectors to prevent information loss.

### 4. Hybrid Scoring Formula
$$S_{\text{final}}(u, m) = 0.5 \cdot \text{Sim}_{\text{content}}(\mathbf{u}, \mathbf{x}_m) + 0.5 \cdot \text{Support}_{\text{peer}}(c_u, m)$$

- **$\text{Sim}_{\text{content}}$**: High-dimensional Cosine Similarity between candidate movie $\mathbf{x}_m$ and user taste centroid $\mathbf{u}$.
- **$\text{Support}_{\text{peer}}$**: Consensus rating score among users in your K-Means peer cohort $c_u$.
- **Strict Monotonic Ordering**: Recommended items are sorted in strictly descending order with detailed score breakdowns.

---

## 📊 Benchmarks & Evaluation

Evaluated on an **offline 80/20 train/test holdout benchmark across 100 real users** with zero data leakage:

| Evaluation Metric | Top-5 ($K=5$) | Top-10 ($K=10$) | Meaning / Interpretation |
|---|:---:|:---:|---|
| **Hit Rate@K** | **84.00%** | **84.00%** | **84 out of 100 users** found a movie they loved in their top recommendations |
| **Precision@K** | **49.60%** | **24.80%** | Nearly **1 in 2 recommended movies** matched actual holdout preferences |
| **Recall@K** | **12.82%** | **12.82%** | Proportion of total test holdout items successfully retrieved |
| **Query Latency** | **< 45 ms** | **< 48 ms** | Sub-50 millisecond inference time per user query |

---

## 🗄️ Database Design

The storage layer is an embedded **SQLite3** relational database (`database/MovieDude.db`):

```text
MovieDude.db
├── Movies_sorted  (6,069 released films, metadata, keywords, ratings)
├── Users_data     (83,911 user ratings, user_rate, liked boolean flag)
└── Users          (User credentials, authentication, timestamps)
```

- **Composite Primary Keys**: `PRIMARY KEY (user_id, movie_id)` ensures data integrity.
- **Atomic Upserts**: `INSERT ... ON CONFLICT DO UPDATE` allows immediate rating updates without table locks.
- **Cache Management**: Database write events atomically evict the in-memory candidate cache, guaranteeing zero staleness.

---

## 💻 CLI Tour

Upon launching the application, you are greeted with an interactive 7-option menu:

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
