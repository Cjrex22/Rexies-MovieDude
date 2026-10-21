# MovieDude 🎬
> **Hybrid Latent Movie Recommendation Engine Powered by Unsupervised K-Means Clustering, 2D PCA & Content-Collaborative Peer Fusion**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-orange.svg)](https://scikit-learn.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-green.svg)](https://www.sqlite.org/)
[![Architecture](https://img.shields.io/badge/Model-Hybrid%20K--Means%20%2B%20PCA-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MovieDude is an end-to-end, machine learning-driven movie recommendation platform that combines **Relational Database Design (SQLite)** with **Unsupervised Machine Learning**. Rather than relying on simple genre lookups, the system partitions both the movie catalog and the user interaction space into latent clusters, performing fine-grained **Cosine Similarity ranking** fused with **Peer Cohort Consensus**.

---

## 🚀 Quick Start (Run in 3 Steps)

### Step 1: Open your terminal in the project folder
```bash
cd movieml
```

### Step 2: Install dependencies (Run once)
```bash
pip install -r requirements.txt
```
*(Recommended: Use a virtual environment: `python3 -m venv .venv && source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows)*

### Step 3: Launch the application
```bash
python3 src/main.py
```
*(On Windows, use `python src/main.py`)*

### 🔑 Pre-Configured Login Credentials:
- **User ID:** `Aryan`
- **Password:** `12345`

---

## 🧠 Machine Learning Architecture

```
[User Interaction / Ratings in SQLite]
               │
               ▼
   [Feature Engineering Layer]
   ├── Multi-Hot Genre Vectors (19 genres)
   ├── L2 Footprint Normalization (Equalizes multi-genre films)
   └── TF-IDF Thematic Keywords
               │
               ├─────────────────────────────────────────┐
               ▼                                         ▼
   [Unsupervised Item Topology]              [Unsupervised User Topology]
   • K-Means (K = 10 Clusters)               • K-Means (K = 6 Cohorts)
   • Centroid: Lloyd's + K-Means++           • Aggregated User Vectors
   • Inertia: 4,194.06                       • Silhouette Score: +0.1358
   • Silhouette: +0.0636                                 │
               │                                         │
               ├────────────────────────┬────────────────┘
               ▼                        ▼
   [2D PCA Visualization Layer]    [Hybrid Scoring Engine]
   • Eigen-decomposition of Cov.   • Quality Gate: r >= 7.5 only
   • Item PC1+PC2: 11.77% var      • Weight: w_i = max(0.5, r_i - 6.5)
   • Strictly for 2D HTML Cartography • S_final = 0.5 * Sim_content + 0.5 * Support_peer
                                    • Franchise Deduplication (Max 1 per series)
                                    • Monotonic Strictly Descending Sort
```

### 1. High-Dimensional Feature Representation
- **Multi-Hot Genre Encoding**: 19 distinct genres encoded into unit vectors.
- **Per-Item L2 Footprint Normalization**: Prevents multi-genre films (e.g., Action+Adventure+Sci-Fi) from artificially overpowering single-genre artistic cinema.
- **TF-IDF Keyword Weighting**: Penalizes ubiquitous words while surfacing distinctive narrative motifs (e.g., *police investigation*, *alibi*, *stockbroker*).

### 2. Dual K-Means Clustering
- **Item Clustering ($K=10$)**: Automatically discovers 10 latent movie categories across 6,000+ films (e.g., Crime/Neo-Noir, Sci-Fi/Dystopia, Animation/Family, Inspiring Biographies).
  - *Inertia (WCSS):* `4,194.06` | *Silhouette Score:* `+0.0636`
- **User Cohort Clustering ($K=6$)**: Groups 600+ users into 6 distinct behavioral taste cohorts based on rating distribution patterns.
  - *User Silhouette Score:* `+0.1358` (Demonstrates clean taste community boundaries).

### 3. Dimensionality Reduction (2D PCA)
- Compresses high-dimensional item and user manifolds into 2D orthogonal coordinates using **Principal Component Analysis (PCA)**.
- Generates an interactive HTML5 visualizer (`reports/movie_clusters_pca.html`) with interactive zoom, centroids, and hover inspection.
- **Key Design Decision**: 2D PCA coordinates are used **strictly for visual inspection**. Recommendation ranking always runs in high-dimensional feature space to preserve 100% of information.

### 4. Hybrid Scoring Formula
$$S_{\text{final}}(u, m) = 0.5 \cdot \text{Sim}_{\text{content}}(\mathbf{u}, \mathbf{x}_m) + 0.5 \cdot \text{Support}_{\text{peer}}(c_u, m)$$
- **Content Similarity**: Cosine similarity between candidate movie vector and the user's weighted taste centroid.
- **Peer Cohort Support**: Normalized rating density from users inside the same K-Means taste cohort.
- Results are sorted in **strict descending order**, clearly displaying `Final Rank Score`, `Content Sim`, and `Peer Cohort Support`.

### 5. Smart Quality Gating & Diversity Safeguards
- **$\ge 7.5$ Threshold Gate**: Only movies rated 7.5 or higher shape the user's taste centroid. Lukewarm or disliked movies are recorded in history but excluded from recommendations.
- **Rating Power Weighting**: $w_i = \max(0.5, r_i - 6.5)$ ensures stronger favorites carry higher mathematical influence (e.g., 8.5 gets double the weight of 7.5).
- **Franchise Deduplication**: Collapses sequels and spin-offs (e.g., *Toy Story 1, 2, 3*) so only the top-scoring entry occupies a recommendation slot.
- **Multilingual Typo-Tolerant Search**: Strips unicode diacritics (`Bāhubali` $\rightarrow$ `Bahubali`), handles phonetic typos (`bhaubali 2`), and presents interactive version pickers for multi-part titles (*Drishyam*, *KGF*).

---

## 📊 Offline Evaluation & Benchmarks

Evaluated on an **offline 80/20 train/test holdout benchmark across 100 real users** with zero data leakage:

| Evaluation Metric | Top-5 ($K=5$) | Top-10 ($K=10$) | Plain-English Meaning |
|---|---|---|---|
| **Hit Rate@K** | **84.00%** | **84.00%** | 84 out of 100 holdout users found a movie they loved in Top-K. |
| **Precision@K** | **49.60%** | **24.80%** | ~1 in 2 recommendations in Top-5 matched true holdout preferences. |
| **Recall@K** | **12.82%** | **12.82%** | Proportion of total holdout items successfully retrieved. |
| **Query Latency** | **< 0.045s** | **< 0.048s** | Sub-50ms inference time per recommendation query. |

---

## 🗄️ Database Architecture (`MovieDude.db`)

The storage layer is backed by an embedded, ACID-compliant **SQLite3** relational database:

```text
MovieDude.db
├── Movies_sorted (6,069 released films, metadata, keywords, ratings)
├── Users_data    (83,911 user ratings, user_rate, liked boolean flag)
└── Users         (User credentials, authentication, timestamps)
```
- **Composite Primary Keys**: Enforces `(user_id, movie_id)` uniqueness.
- **Atomic Upserts**: Uses `INSERT ... ON CONFLICT(user_id, movie_id) DO UPDATE` for instant, atomic profile saves without restarting the application.
- **In-Memory Cache Invalidation**: Ratings changes automatically trigger cache eviction, guaranteeing zero-staleness recommendations.

---

## 🖥️ Interactive CLI Features

```text
1 : Discover Movies Based on Your Activities (User Taste Clustering)
2 : Find Similar Movies by Given Movie Title (Item Clustering & Latent Space)
3 : Rate or Log a Movie
4 : View My Watch & Rating History
5 : Explore Movie Clusters & PCA Visualizer (Interactive Latent Map)
6 : Switch User / Log Out
7 : Exit
```

- **Option 1**: Generates your personalized Top 10 recommendations with detailed match rationales and dual content/peer score breakdowns. Includes a diverse cold-start fallback for new users.
- **Option 2**: Item-based similarity search inside the seed movie's K-Means cluster with franchise deduplication.
- **Option 3**: Log any movie with a rating (1.0 to 10.0) and like/dislike flag with instant database persistence.
- **Option 4**: View your watch history with color-coded seed tags (`[Seed: Rated ≥ 7.5]` vs `[Excluded: Rated < 7.5]`).
- **Option 5**: Displays live K-Means metrics and automatically launches the interactive 2D PCA cluster map in your default web browser.

---

## 📁 Project Directory Structure

```text
movieml/
├── database/
│   └── MovieDude.db                     # SQLite database (6,069 movies, 83k ratings)
├── reports/
│   ├── movie_clusters_pca.html          # Interactive 2D PCA browser visualizer
│   └── movie_clusters_pca.svg           # Vector PCA cluster graphic
├── src/
│   ├── main.py                          # CLI application entry point
│   ├── clustering_benchmark.py          # Full model evaluation & benchmark suite
│   ├── evaluation.py                    # Holdout cross-validation module
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── clustering.py                # Dual K-Means, 2D PCA & franchise deduplication
│   │   ├── engine.py                    # Hybrid recommendation engine
│   │   ├── find_by_title.py             # Diacritic-safe search & version disambiguation
│   │   ├── find_user_interests.py       # User taste vectorization & 7.5 threshold gating
│   │   └── watched_movies.py            # User history retrieval
│   └── scripts/
│       └── login.py                     # User authentication
├── requirements.txt                     # Dependency specifications
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).
