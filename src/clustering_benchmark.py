import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from modules.clustering import get_clustering_model
from evaluation import evaluate_clustering_and_pca, evaluate_holdout_recommendations

def run_clustering_benchmark():
    db_path = os.path.join(BASE_DIR, "database", "MovieDude.db")

    print("=" * 88)
    print("      MOVIEDUDE UNSUPERVISED LEARNING & RECOMMENDATION BENCHMARK (K-MEANS & PCA)")
    print("=" * 88)

    t0 = time.time()
    model = get_clustering_model(db_path)
    print(f"Loaded and initialized model in {time.time() - t0:.2f}s")
    print(f"Total Unique Movies in Dataset: {len(model.df):,}")
    print(f"Total Active Users in Dataset:  {len(model.user_ids_list):,}")

    # 1. Clustering & PCA Evaluation
    unsup = evaluate_clustering_and_pca(model)

    print("\n[1/4] ITEM CLUSTERING & 2D PCA DIMENSIONALITY REDUCTION")
    print("-" * 65)
    print(f"K-Means Item Clusters (K={unsup['item_clusters']}):")
    print(f"  • Inertia (WCSS):          {unsup['item_inertia']:,.2f}")
    print(f"  • Silhouette Score:        {unsup['item_silhouette']:.4f}")
    print(f"  • Cluster Sizes (Min/Max): {unsup['item_dist']['min']} / {unsup['item_dist']['max']} (Mean: {unsup['item_dist']['mean']:.1f}, Std: {unsup['item_dist']['std']:.1f})")
    i_evr = unsup['item_pca_evr']
    print(f"  • 2D PCA Explained Variance: PC1: {i_evr[0]*100:.2f}%, PC2: {i_evr[1]*100:.2f}% (Total: {sum(i_evr)*100:.2f}%)")

    print("\n[2/4] USER TASTE COHORT CLUSTERING & 2D PCA")
    print("-" * 65)
    print(f"K-Means User Cohorts (K={unsup['user_clusters']}):")
    print(f"  • Inertia (WCSS):          {unsup['user_inertia']:,.2f}")
    print(f"  • Silhouette Score:        {unsup['user_silhouette']:.4f}")
    print(f"  • Cohort Sizes (Min/Max):  {unsup['user_dist']['min']} / {unsup['user_dist']['max']} (Mean: {unsup['user_dist']['mean']:.1f}, Std: {unsup['user_dist']['std']:.1f})")
    u_evr = unsup['user_pca_evr']
    if u_evr:
        print(f"  • 2D PCA Explained Variance: PC1: {u_evr[0]*100:.2f}%, PC2: {u_evr[1]*100:.2f}% (Total: {sum(u_evr)*100:.2f}%)")

    print("\n" + "=" * 88)
    print("                    DISCOVERED LATENT TASTE CLUSTERS")
    print("=" * 88)
    for c in range(model.n_clusters):
        c_df = model.df[model.df['cluster'] == c]
        top_movies = c_df.sort_values(by='rating_count', ascending=False)['title'].head(3).tolist()
        pct = len(c_df) / len(model.df) * 100
        print(f"▶ Cluster #{c}: {model.cluster_names[c]}")
        print(f"  • Size: {len(c_df):,} titles ({pct:.1f}%) | Exemplars: {', '.join(top_movies)}")

    print("\n" + "=" * 88)
    print("               USER TASTE COHORTS (PEER INTERACTION CLUSTERS)")
    print("=" * 88)
    for c in range(model.n_user_clusters):
        u_count = sum(1 for _, coh in model.user_cohort_assignments.items() if coh == c)
        pct = u_count / len(model.user_ids_list) * 100
        print(f"▶ Cohort #{c}: {model.user_cluster_names.get(c, f'Cohort {c}')}")
        print(f"  • Size: {u_count} users ({pct:.1f}%)")

    # 2. Holdout Recommendation Performance (Leak-Free, Pre-Split)
    print("\n" + "=" * 88)
    print("      HOLDOUT RECOMMENDATION EVALUATION (80/20 TRAIN/TEST SPLIT, ZERO LEAKAGE)")
    print("=" * 88)
    rec_metrics = evaluate_holdout_recommendations(db_path, k_list=[5, 10], sample_size=100)
    for k, res in rec_metrics.items():
        print(f"▶ Top-{k} Metrics ({res['evaluated_users']} holdout users):")
        print(f"  • Hit Rate@{k}:   {res['hit_rate']*100:.2f}%")
        print(f"  • Precision@{k}:  {res['precision']*100:.2f}%")
        print(f"  • Recall@{k}:     {res['recall']*100:.2f}%")

    # 3. Seed Item Tests with Franchise Deduplication Verification
    print("\n" + "=" * 88)
    print("            ITEM-BASED RECOMMENDATIONS (COSINE SIMILARITY ON CLUSTERS)")
    print("=" * 88)
    test_queries = [
        ("La La Land", 313369),
        ("Toy Story", 862),
        ("The Dark Knight", 155),
        ("Whiplash", 244786),
        ("The Notebook", 11036)
    ]

    for title, mid in test_queries:
        c_info = model.get_movie_cluster(mid)
        if not c_info:
            continue
        print(f"\nSeed: '{title}' -> Mapped to Item Cluster #{c_info['cluster_id']}: {c_info['cluster_name']} (PCA: x={c_info['pca_x']:.3f}, y={c_info['pca_y']:.3f})")
        recs = model.recommend_by_movie_cluster(mid, top_n=10)
        from modules.clustering import get_franchise_key
        fkeys = [get_franchise_key(r['title']) for r in recs]
        key_counts = Counter(fkeys)
        has_dups = any(cnt > 1 for cnt in key_counts.values())
        print(f"  Franchise Deduplication Check: {'❌ DUPLICATES FOUND: ' + str([k for k, v in key_counts.items() if v > 1]) if has_dups else '✅ CLEAN (Max 1 movie per franchise)'}")
        for idx, r in enumerate(recs, 1):
            print(f"   {idx:2d}. {r['title']} ({r.get('year')}) • Cosine Sim: {r['score']:.4f} [FKey: '{get_franchise_key(r['title'])}']")

    # 4. User-Based Recommendations & Ranking Consistency
    print("\n" + "=" * 88)
    print("         USER-BASED RECOMMENDATIONS & RANKING CONSISTENCY (ARYAN)")
    print("=" * 88)
    from modules import find_user_interests
    aryan_interests = find_user_interests(db_path, "Aryan")
    aryan_recs = model.recommend_by_user_cluster(aryan_interests, top_n=10)
    scores = [r['score'] for r in aryan_recs]
    is_sorted_desc = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"Ranking Score Strict Descending Check: {'✅ PASS (Strictly descending)' if is_sorted_desc else '❌ FAIL (Not descending!)'}")
    for idx, r in enumerate(aryan_recs, 1):
        print(f"   {idx:2d}. {r['title']} ({r.get('year')}) | Final Score: {r['score']:.4f} (Content Sim: {r.get('content_similarity', 0):.4f}, Peer Support: {r.get('peer_support', 0):.4f})")

    # 5. Cold-Start / New User Recommendations
    print("\n" + "=" * 88)
    print("            COLD-START / UNSEEDED USER DIVERSE EXEMPLARS")
    print("=" * 88)
    cold_recs = model.recommend_by_user_cluster({}, top_n=10)
    for idx, r in enumerate(cold_recs, 1):
        print(f"   {idx:2d}. {r['title']} ({r.get('year')}) • Cluster #{r['cluster_id']}: {r['cluster_name']}")

    # 6. Generate Visualizations
    report_html = os.path.join(BASE_DIR, "reports", "movie_clusters_pca.html")
    report_svg = os.path.join(BASE_DIR, "reports", "movie_clusters_pca.svg")
    model.generate_html_visualization(report_html)
    model.generate_svg_visualization(report_svg)

    print("\n" + "=" * 88)
    print("               DUAL PCA DIMENSIONALITY REDUCTION VISUALIZATIONS")
    print("=" * 88)
    print(f"1. Interactive HTML5 Visualizer (Dual-view Item & User Map):\n   {report_html}")
    print(f"2. Vector SVG Graphic:\n   {report_svg}")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    run_clustering_benchmark()
