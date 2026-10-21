"""
MovieDude Comprehensive Recommendation & Clustering Evaluation Suite
Implements:
1. Silhouette Score (Item & User Clustering)
2. Inertia (Within-Cluster Sum of Squares)
3. Cluster Size Distribution (Min, Max, Mean, Std)
4. PCA Explained Variance Ratios (Items & Users)
5. Precision@K, Recall@K, and Hit Rate@K using 80/20 train/test holdout validation on authentic user interaction data.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modules.clustering import get_clustering_model

def evaluate_clustering_and_pca(model):
    """Computes and formats unsupervised metrics for both item and user clustering."""
    metrics = model.get_cluster_metrics()

    # Item cluster distribution
    item_cluster_counts = model.df['cluster'].value_counts().sort_index().tolist()
    item_dist = {
        "min": int(np.min(item_cluster_counts)),
        "max": int(np.max(item_cluster_counts)),
        "mean": float(np.mean(item_cluster_counts)),
        "std": float(np.std(item_cluster_counts)),
        "counts": item_cluster_counts
    }

    # User cluster distribution
    user_cohort_counts = pd.Series(list(model.user_cohort_assignments.values())).value_counts().sort_index().tolist()
    user_dist = {
        "min": int(np.min(user_cohort_counts)) if user_cohort_counts else 0,
        "max": int(np.max(user_cohort_counts)) if user_cohort_counts else 0,
        "mean": float(np.mean(user_cohort_counts)) if user_cohort_counts else 0.0,
        "std": float(np.std(user_cohort_counts)) if user_cohort_counts else 0.0,
        "counts": user_cohort_counts
    }

    return {
        "item_clusters": metrics["n_item_clusters"],
        "item_inertia": metrics["item_inertia"],
        "item_silhouette": metrics["item_silhouette_score"],
        "item_dist": item_dist,
        "item_pca_evr": metrics["pca_explained_variance_ratio"],
        "user_clusters": metrics["n_user_clusters"],
        "user_inertia": metrics["user_inertia"],
        "user_silhouette": metrics["user_silhouette_score"],
        "user_dist": user_dist,
        "user_pca_evr": metrics.get("user_pca_explained_variance_ratio", []),
        "total_movies": metrics["total_movies"],
        "total_users": metrics["total_users"]
    }

def evaluate_holdout_recommendations(db_path: str, model=None, k_list: list = [5, 10], sample_size: int = 100, random_state: int = 42):
    """
    Performs rigorous, leak-free 80/20 train/test holdout evaluation:
    1. Pre-splits user interaction data BEFORE fitting any rating-dependent components.
    2. Held-out test positive interactions are completely excluded from:
       - Item collaborative feature matrix (TruncatedSVD)
       - User preference profiling and K-Means taste cohort clustering
       - Cohort peer ratings and consensus scores
    3. During testing, candidate retrieval strictly excludes items in the user's training set,
       allowing held-out positive ground-truth movies to be genuinely discovered.
    4. Computes Precision@K, Recall@K, and Hit Rate@K across 100 active test users.
    """
    from modules.clustering import MovieClusterModel

    conn = sqlite3.connect(db_path)
    df_u = pd.read_sql("SELECT user_id, movie_id, user_rate, liked FROM Users_data", conn)
    conn.close()

    # Identify eligible users with at least 20 positive ratings
    df_pos = df_u[(df_u['user_rate'] >= 7.0) | (df_u['liked'] == 1)]
    user_counts = df_pos['user_id'].value_counts()
    eligible_users = sorted(user_counts[user_counts >= 20].index.tolist())

    rng = np.random.RandomState(random_state)
    sample_users = rng.choice(eligible_users, size=min(sample_size, len(eligible_users)), replace=False)
    sample_user_set = set(sample_users)

    # Construct clean train set by withholding 20% positive interactions for the sample users
    train_rows = []
    user_eval_data = {}

    # All interactions for non-evaluated users remain in training
    non_eval_df = df_u[~df_u['user_id'].isin(sample_user_set)]
    train_rows.append(non_eval_df)

    for u in sample_users:
        u_all = df_u[df_u['user_id'] == u].copy()
        u_pos = u_all[(u_all['user_rate'] >= 7.0) | (u_all['liked'] == 1)]

        n_pos = len(u_pos)
        n_train = max(1, int(n_pos * 0.8))

        # Deterministic shuffle
        u_pos_shuffled = u_pos.sample(frac=1.0, random_state=random_state)
        train_pos = u_pos_shuffled.iloc[:n_train]
        test_pos = u_pos_shuffled.iloc[n_train:]

        train_pos_mids = train_pos['movie_id'].tolist()
        test_pos_mids = set(test_pos['movie_id'].tolist())

        # Training interactions exclude held-out test movies
        u_train_all = u_all[~u_all['movie_id'].isin(test_pos_mids)]
        train_watched_mids = list(set(u_train_all['movie_id'].tolist()))

        train_rows.append(u_train_all)
        user_eval_data[u] = {
            "train_pos_mids": train_pos_mids,
            "train_watched_mids": train_watched_mids,
            "test_pos_mids": test_pos_mids
        }

    df_u_train = pd.concat(train_rows, ignore_index=True)

    # Initialize and fit model strictly on training interactions (ZERO TEST LEAKAGE)
    eval_model = MovieClusterModel(db_path, ratings_df=df_u_train, random_state=random_state)

    results = {}
    for k in k_list:
        precisions = []
        recalls = []
        hits = []

        for u in sample_users:
            u_info = user_eval_data[u]
            test_pos_mids = u_info["test_pos_mids"]

            if not test_pos_mids:
                continue

            user_interests = {
                "positive_movie_ids": u_info["train_pos_mids"],
                "watched_movie_ids": u_info["train_watched_mids"]
            }

            # Candidate retrieval excludes user's training items
            recs = eval_model.recommend_by_user_cluster(user_interests, top_n=k, filter_watched=True)
            rec_mids = [r['movie_id'] for r in recs]

            hit_count = len(set(rec_mids).intersection(test_pos_mids))
            precisions.append(hit_count / k)
            recalls.append(hit_count / len(test_pos_mids))
            hits.append(1 if hit_count > 0 else 0)

        results[k] = {
            "hit_rate": float(np.mean(hits)) if hits else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "evaluated_users": len(precisions)
        }

    return results

def run_full_evaluation():
    db_path = os.path.join(BASE_DIR, "database", "MovieDude.db")
    model = get_clustering_model(db_path)

    print("=" * 86)
    print("      MOVIEDUDE RIGOROUS EVALUATION BENCHMARK (REAL DATASETS)")
    print("=" * 86)

    # 1. Unsupervised Clustering & PCA Metrics
    unsup = evaluate_clustering_and_pca(model)

    print("\n[1/3] ITEM & USER CLUSTERING METRICS (K-MEANS)")
    print("-" * 65)
    print(f"Item Clusters (K={unsup['item_clusters']}):")
    print(f"  • Inertia (WCSS):          {unsup['item_inertia']:,.2f}")
    print(f"  • Silhouette Score:        {unsup['item_silhouette']:.4f}")
    print(f"  • Cluster Sizes (Min/Max): {unsup['item_dist']['min']} / {unsup['item_dist']['max']} (Mean: {unsup['item_dist']['mean']:.1f}, Std: {unsup['item_dist']['std']:.1f})")

    print(f"\nUser Taste Cohorts (K={unsup['user_clusters']}):")
    print(f"  • Inertia (WCSS):          {unsup['user_inertia']:,.2f}")
    print(f"  • Silhouette Score:        {unsup['user_silhouette']:.4f}")
    print(f"  • Cohort Sizes (Min/Max):  {unsup['user_dist']['min']} / {unsup['user_dist']['max']} (Mean: {unsup['user_dist']['mean']:.1f}, Std: {unsup['user_dist']['std']:.1f})")

    print("\n[2/3] DIMENSIONALITY REDUCTION METRICS (2D PCA)")
    print("-" * 65)
    i_evr = unsup['item_pca_evr']
    print(f"Movie Feature PCA Variance: PC1: {i_evr[0]*100:.2f}%, PC2: {i_evr[1]*100:.2f}% (Total: {sum(i_evr)*100:.2f}%)")
    u_evr = unsup['user_pca_evr']
    if u_evr:
        print(f"User Profile PCA Variance:  PC1: {u_evr[0]*100:.2f}%, PC2: {u_evr[1]*100:.2f}% (Total: {sum(u_evr)*100:.2f}%)")

    # 2. Holdout Recommendation Evaluation
    print("\n[3/3] HOLDOUT RECOMMENDATION EVALUATION (80/20 TRAIN/TEST SPLIT)")
    print("-" * 65)
    print("Evaluating held-out user preferences against 5,915 catalogue items...")
    rec_metrics = evaluate_holdout_recommendations(db_path, model, k_list=[5, 10], sample_size=100)

    for k, res in rec_metrics.items():
        print(f"\n▶ Recommendation Metrics @ K={k} ({res['evaluated_users']} holdout users):")
        print(f"  • Hit Rate@{k}:   {res['hit_rate']*100:.2f}%")
        print(f"  • Precision@{k}:  {res['precision']*100:.2f}%")
        print(f"  • Recall@{k}:     {res['recall']*100:.2f}%")

    print("\n" + "=" * 86)
    print("✅ All evaluation metrics computed on authentic dataset interactions.")
    print("=" * 86 + "\n")

if __name__ == "__main__":
    run_full_evaluation()
