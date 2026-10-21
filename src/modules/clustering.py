import os
import re
import json
import sqlite3
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize, MinMaxScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Global singleton cache for low-latency interactive execution
_GLOBAL_CLUSTER_MODEL = None

GENERIC_KW_STOPWORDS = {
    'film', 'short story', 'independent film', 'duringcreditsstinger', 'aftercreditsstinger',
    'based on novel or book', 'based on short film', 'based on toy', 'based on tv series',
    'woman director', 'los angeles', 'california', 'new york city', 'london', 'paris', 'france',
    'hollywood', 'movie set', 'sequel', 'prequel', 'remake', 'scene', 'title'
}

def get_franchise_key(title: str) -> str:
    """
    Extracts a robust franchise identifier from a movie title.
    Handles:
    - Subtitles after ':', '-', '–', '/'
    - Sequel numbers, Roman numerals (II, III, IV, etc.), 'part', 'chapter', 'vol', 'edition', '3d'
    - Stopwords ('the', 'a', 'an', 'of')
    - Prevents false-grouping of distinct movies sharing common first words (e.g. 'American Beauty' vs 'American Psycho')
      by keeping up to 2 significant root words.
    """
    if not title:
        return ""
    t = re.sub(r'\(.*?\)|\[.*?\]', '', title)
    t = re.split(r'[:\-\–\/]', t)[0]
    t = re.sub(r'[^\w\s]', ' ', t).lower()
    words = t.split()

    STOPWORDS = {'the', 'a', 'an', 'and', 'of', 'in', 'to', 'for', 'on', 'with', 'by'}
    SEQUEL_WORDS = {
        'part', 'vol', 'volume', 'chapter', 'season', 'episode', 'edition',
        'movie', 'film', '3d', 'reloaded', 'revolutions', 'resurrections',
        'returns', 'forever', 'begins', 'rises', 'awakens', 'legacy', 'infinity'
    }
    ROMAN_NUMERALS = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}

    cleaned = []
    for w in words:
        if w in SEQUEL_WORDS or w in ROMAN_NUMERALS or w.isdigit():
            break
        if w not in STOPWORDS:
            cleaned.append(w)

    if not cleaned:
        cleaned = [w for w in words if not w.isdigit()]

    key = " ".join(cleaned[:2]) if len(cleaned) >= 2 else (" ".join(cleaned) if cleaned else title.strip().lower())
    return key

# Backwards compatibility alias
clean_movie_title_root = get_franchise_key

class MovieClusterModel:
    """
    Comprehensive Unsupervised Machine Learning Recommendation & Visualization Engine:
    1. Multi-modal Feature Engineering:
       - Genre Profiles: Multi-hot with smooth Inverse Document Frequency (IDF) weighting.
       - Thematic Content: Cleaned plot keywords, directors, and stars via Sublinear TF-IDF.
       - User Collaborative Signal: Latent co-rating patterns (TruncatedSVD) from Users_data.
       - Metadata Attributes: Scaled release era and Bayesian quality scores.
       - Balanced Composite Matrix: Normalized multi-block representation.
    2. Item Clustering: K-Means (K=10) on composite feature space.
    3. User Clustering: K-Means (K=6) on user genre & cluster preference interaction profiles.
    4. Dimensionality Reduction: Genuine PCA for both Movies and Users to 2D latent space.
    5. Cosine Similarity Ranking: Candidate selection via clusters, ranked strictly by cosine similarity.
    6. Dual Visualizer: Standalone Interactive HTML5 Dashboard (Item + User toggles) + Vector SVG.
    """

    CLUSTER_PALETTE = [
        "#10B981",  # Emerald (Crime / Thriller / Action)
        "#3B82F6",  # Blue (Family / Animation)
        "#EC4899",  # Pink (Comedy / Romance / Music)
        "#F59E0B",  # Amber (War / History / Biography)
        "#8B5CF6",  # Purple (Mystery / Psychological Drama)
        "#06B6D4",  # Cyan (Sci-Fi / Space / Cyberpunk)
        "#EF4444",  # Red (Horror / Supernatural / Slasher)
        "#6366F1",  # Indigo (Dramatic Fiction / Arthouse)
        "#14B8A6",  # Teal (Adventure / Fantasy)
        "#F97316"   # Orange (Action / Superhero)
    ]

    USER_PALETTE = [
        "#38BDF8",  # Sky Blue
        "#A855F7",  # Purple
        "#34D399",  # Mint Green
        "#F43F5E",  # Rose
        "#FBBF24",  # Amber
        "#818CF8"   # Indigo
    ]

    def __init__(self, db_path: str, n_item_clusters: int = 10, n_user_clusters: int = 6, random_state: int = 42, ratings_df: pd.DataFrame = None):
        self.db_path = db_path
        self.n_clusters = n_item_clusters
        self.n_user_clusters = n_user_clusters
        self.random_state = random_state
        self.ratings_df = ratings_df

        # Movie data and representations
        self.df = None
        self.id_to_idx = {}
        self.X_composite = None
        self.kmeans = None
        self.pca = None
        self.pca_evr = []
        self.inertia = 0.0
        self.silhouette = 0.0
        self.cluster_names = {}
        self.cluster_colors = {}

        # User clustering representations
        self.user_df = None
        self.U_feat = None
        self.user_kmeans = None
        self.user_pca = None
        self.user_pca_evr = []
        self.user_inertia = 0.0
        self.user_silhouette = 0.0
        self.user_cohort_assignments = {}
        self.user_cluster_names = {}
        self.cohort_peer_ratings = {}  # cohort_id -> dict of movie_id -> aggregated positive score

        self._fit_items()
        self._fit_users()

    def _fit_items(self):
        """Builds balanced multi-modal movie feature matrix, fits K-Means and PCA."""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql(
            "SELECT movie_id, title, genres, keywords, director, stars, studio, country, release_year, rating, rating_count, final_score FROM Movies_sorted",
            conn
        )
        if self.ratings_df is not None:
            df_u = self.ratings_df[['user_id', 'movie_id', 'user_rate']].copy()
        else:
            df_u = pd.read_sql("SELECT user_id, movie_id, user_rate FROM Users_data", conn)
        conn.close()

        # Deduplicate titles & IDs
        self.df = self.df.drop_duplicates(subset=['title']).reset_index(drop=True)
        self.df = self.df.drop_duplicates(subset=['movie_id']).reset_index(drop=True)
        self.id_to_idx = {mid: i for i, mid in enumerate(self.df['movie_id'])}

        # -------------------------------------------------------------
        # Feature Block 1: Genres with Smooth IDF Weighting
        # -------------------------------------------------------------
        genres_list = [[g.strip().lower() for g in str(row['genres']).split(',') if g.strip()] for _, row in self.df.iterrows()]
        mlb = MultiLabelBinarizer()
        X_genre_raw = mlb.fit_transform(genres_list).astype(float)
        self.genre_classes_ = mlb.classes_
        df_genre = pd.DataFrame(X_genre_raw, columns=mlb.classes_)
        genre_counts = df_genre.sum(axis=0)
        # Niche genres (e.g. Music, Animation, Western) receive higher specificity than ubiquitous Drama/Comedy
        genre_idf = np.log((len(self.df) + 1.0) / (genre_counts + 1.0)) + 1.0
        X_genre = normalize(X_genre_raw * genre_idf.values, norm='l2', axis=1)

        # -------------------------------------------------------------
        # Feature Block 2: Cleaned Thematic Keywords & Metadata TF-IDF
        # -------------------------------------------------------------
        def clean_keywords(kw_str):
            if not kw_str:
                return ''
            tokens = [k.strip().lower() for k in str(kw_str).split(',') if k.strip()]
            cleaned = []
            for t in tokens:
                if t not in GENERIC_KW_STOPWORDS:
                    cleaned.append(re.sub(r'[\s\-]+', '_', t))
            return ' '.join(cleaned)

        clean_kw_series = self.df['keywords'].apply(clean_keywords)

        def make_soup(row, kw_clean):
            kw = kw_clean
            dir_ = re.sub(r'[\s\-]+', '_', str(row.get('director') or '').lower().strip()) if row.get('director') else ''
            stars = ' '.join([re.sub(r'[\s\-]+', '_', s.lower().strip()) for s in str(row.get('stars') or '').split(',') if s.strip()]) if row.get('stars') else ''
            return f"{kw} {kw} {dir_} {stars}".strip()

        soups = [make_soup(row, clean_kw_series.iloc[i]) for i, (_, row) in enumerate(self.df.iterrows())]
        tfidf = TfidfVectorizer(max_features=2500, stop_words='english', sublinear_tf=True, min_df=2)
        X_content = normalize(tfidf.fit_transform(soups).toarray(), norm='l2', axis=1)

        # -------------------------------------------------------------
        # Feature Block 3: Collaborative User Interaction Patterns (SVD)
        # -------------------------------------------------------------
        user_ids = sorted(df_u['user_id'].unique())
        u_map = {u: i for i, u in enumerate(user_ids)}
        df_u_filt = df_u[df_u['movie_id'].isin(self.id_to_idx)]
        rows = [self.id_to_idx[mid] for mid in df_u_filt['movie_id']]
        cols = [u_map[uid] for uid in df_u_filt['user_id']]
        data = df_u_filt['user_rate'].values
        R_sparse = csr_matrix((data, (rows, cols)), shape=(len(self.df), len(user_ids)))
        svd_cf = TruncatedSVD(n_components=25, random_state=self.random_state)
        X_collab_raw = svd_cf.fit_transform(R_sparse)
        X_collab = normalize(X_collab_raw, norm='l2', axis=1)

        # -------------------------------------------------------------
        # Feature Block 4: Metadata (Era & Quality Scores)
        # -------------------------------------------------------------
        scaler = MinMaxScaler()
        meta_vals = np.column_stack([
            self.df['release_year'].fillna(2000).values,
            self.df['final_score'].fillna(0.5).values
        ])
        X_meta = normalize(scaler.fit_transform(meta_vals), norm='l2', axis=1)

        # -------------------------------------------------------------
        # Balanced Composite Matrix
        # Weights: Genre (0.35), Content (0.40), Collaborative (0.20), Meta (0.05)
        # -------------------------------------------------------------
        self.X_composite = np.hstack([
            0.35 * X_genre,
            0.40 * X_content,
            0.20 * X_collab,
            0.05 * X_meta
        ])
        self.X_composite = normalize(self.X_composite, norm='l2', axis=1)

        # -------------------------------------------------------------
        # K-Means Item Clustering
        # -------------------------------------------------------------
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_composite)
        self.inertia = float(self.kmeans.inertia_)
        self.silhouette = float(silhouette_score(self.X_composite, self.df['cluster'], sample_size=2000, random_state=self.random_state))

        # -------------------------------------------------------------
        # PCA Dimensionality Reduction to 2D
        # -------------------------------------------------------------
        self.pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = self.pca.fit_transform(self.X_composite)
        self.df['pca_x'] = X_pca[:, 0]
        self.df['pca_y'] = X_pca[:, 1]
        self.pca_evr = [float(v) for v in self.pca.explained_variance_ratio_]

        # Semantic cluster naming
        self._name_clusters()

        # Popularity score for tie-breaking
        max_votes = self.df['rating_count'].max()
        self.df['pop'] = np.log1p(self.df['rating_count'].fillna(0)) / np.log1p(max_votes) if max_votes > 0 else 1.0

    def _name_clusters(self):
        """Derives human-readable thematic names for each K-Means item cluster."""
        for c in range(self.n_clusters):
            c_df = self.df[self.df['cluster'] == c]
            genres = [g.strip() for g in c_df['genres'].dropna().str.split(',').explode() if g.strip()]
            top_g = [g for g, _ in Counter(genres).most_common(2)]

            keywords = [k.strip() for k in c_df['keywords'].dropna().str.split(',').explode() if k.strip()]
            top_k = [
                k.title() for k, _ in Counter(keywords).most_common(3)
                if k.lower() not in GENERIC_KW_STOPWORDS and k.lower() not in ('film', 'scene')
            ]

            name_parts = top_g.copy()
            if top_k and len(name_parts) < 3:
                name_parts.append(top_k[0])

            cluster_title = " & ".join(name_parts) if name_parts else f"Cluster {c}"
            self.cluster_names[c] = cluster_title
            self.cluster_colors[c] = self.CLUSTER_PALETTE[c % len(self.CLUSTER_PALETTE)]

        self.df['cluster_name'] = self.df['cluster'].map(self.cluster_names)
        self.df['cluster_color'] = self.df['cluster'].map(self.cluster_colors)

    def _fit_users(self):
        """
        Builds user preference profiles and clusters users into taste cohorts via K-Means.
        Derives genuine 2D PCA user coordinates for visualization.
        """
        if self.ratings_df is not None:
            ud = self.ratings_df[['user_id', 'movie_id', 'user_rate', 'liked']].copy()
        else:
            conn = sqlite3.connect(self.db_path)
            ud = pd.read_sql("SELECT user_id, movie_id, user_rate, liked FROM Users_data", conn)
            conn.close()

        if ud.empty:
            return

        id_to_cluster = dict(zip(self.df['movie_id'], self.df['cluster']))
        id_to_genre = dict(zip(self.df['movie_id'], self.df['genres']))

        # Focus on positive interactions (rating >= 7.0 or liked == 1)
        ud_pos = ud[(ud['user_rate'] >= 7.0) | (ud['liked'] == 1)].copy()
        ud_pos['item_cluster'] = ud_pos['movie_id'].map(id_to_cluster)
        ud_pos = ud_pos.dropna(subset=['item_cluster'])
        ud_pos['item_cluster'] = ud_pos['item_cluster'].astype(int)

        # 1. User Genre Preference Vector (dimension: len(genre_classes_))
        ud_pos['genres'] = ud_pos['movie_id'].map(id_to_genre).fillna('')
        ud_pos['genre_list'] = ud_pos['genres'].apply(lambda x: [g.strip().lower() for g in str(x).split(',') if g.strip()])
        user_genre_df = ud_pos.explode('genre_list').groupby(['user_id', 'genre_list']).size().unstack(fill_value=0)
        for g in self.genre_classes_:
            if g not in user_genre_df.columns:
                user_genre_df[g] = 0
        user_genre_df = user_genre_df[self.genre_classes_]
        user_genre_norm = normalize(user_genre_df.values, norm='l2', axis=1)

        # 2. User Item Cluster Affinity Vector (dimension: n_clusters)
        user_cluster_df = pd.crosstab(ud_pos['user_id'], ud_pos['item_cluster'])
        for col in range(self.n_clusters):
            if col not in user_cluster_df.columns:
                user_cluster_df[col] = 0
        user_cluster_df = user_cluster_df[sorted(user_cluster_df.columns)]
        user_cluster_norm = normalize(user_cluster_df.values, norm='l2', axis=1)

        # Composite User Feature Matrix
        self.U_feat = np.hstack([0.60 * user_genre_norm, 0.40 * user_cluster_norm])
        self.U_feat = normalize(self.U_feat, norm='l2', axis=1)
        self.user_ids_list = user_genre_df.index.tolist()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids_list)}

        k_users = min(self.n_user_clusters, len(self.user_ids_list))
        if k_users >= 2:
            self.user_kmeans = KMeans(n_clusters=k_users, random_state=self.random_state, n_init=10)
            user_labels = self.user_kmeans.fit_predict(self.U_feat)
            self.user_cohort_assignments = dict(zip(self.user_ids_list, user_labels))
            self.user_inertia = float(self.user_kmeans.inertia_)
            self.user_silhouette = float(silhouette_score(self.U_feat, user_labels))

            # 3. User PCA Dimensionality Reduction to 2D
            self.user_pca = PCA(n_components=2, random_state=self.random_state)
            U_pca = self.user_pca.fit_transform(self.U_feat)
            self.user_pca_evr = [float(v) for v in self.user_pca.explained_variance_ratio_]

            # Build user points table
            user_total_ratings = ud.groupby('user_id').size().to_dict()
            user_top_genre = user_genre_df.idxmax(axis=1).to_dict()
            self.user_points = []
            for idx, uid in enumerate(self.user_ids_list):
                cohort_id = int(user_labels[idx])
                self.user_points.append({
                    "id": uid,
                    "c": cohort_id,
                    "px": round(float(U_pca[idx, 0]), 4),
                    "py": round(float(U_pca[idx, 1]), 4),
                    "count": user_total_ratings.get(uid, 0),
                    "top_genre": user_top_genre.get(uid, "Mixed").title()
                })

            # Derive cohort names
            for cohort_id in range(k_users):
                cohort_mask = (user_labels == cohort_id)
                top_genre_idx = user_genre_norm[cohort_mask].mean(axis=0).argmax()
                top_genre_name = self.genre_classes_[top_genre_idx].title()
                top_cluster_idx = user_cluster_norm[cohort_mask].mean(axis=0).argmax()
                cluster_label = self.cluster_names.get(top_cluster_idx, f"Cluster {top_cluster_idx}").split('&')[0].strip()
                self.user_cluster_names[cohort_id] = f"{top_genre_name} & {cluster_label} Cohort"

            # Precompute cohort peer rating support for rapid candidate scoring
            self.cohort_peer_ratings = {}
            for cohort_id in range(k_users):
                cohort_users = [u for u, c in self.user_cohort_assignments.items() if c == cohort_id]
                c_ratings = ud_pos[ud_pos['user_id'].isin(cohort_users)]
                agg = c_ratings.groupby('movie_id').agg(
                    pos_count=('user_rate', 'count'),
                    avg_rate=('user_rate', 'mean')
                )
                max_pos = agg['pos_count'].max() if not agg.empty else 1
                self.cohort_peer_ratings[cohort_id] = {
                    mid: float((row['pos_count'] / max_pos) * 0.7 + (row['avg_rate'] / 10.0) * 0.3)
                    for mid, row in agg.iterrows()
                }

    def get_movie_cluster(self, movie_id: int) -> dict:
        """Returns the cluster ID, name, color, and genuine 2D PCA coordinates of a movie."""
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return {}

        if movie_id not in self.id_to_idx:
            return {}
        idx = self.id_to_idx[movie_id]
        row = self.df.iloc[idx]
        return {
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "cluster_id": int(row["cluster"]),
            "cluster_name": row["cluster_name"],
            "pca_x": float(row["pca_x"]),
            "pca_y": float(row["pca_y"])
        }

    def recommend_by_movie_cluster(self, movie_id: int, top_n: int = 10, filter_watched_ids: list = None) -> list[dict]:
        """
        Item Clustering Recommendation (Option 2):
        1. Identifies the query movie's K-Means item cluster as primary candidate space.
        2. Retrieves candidate movies within that cluster (and closest neighbor if small).
        3. Computes genuine Cosine Similarity on preprocessed multi-modal feature vectors.
        4. Strictly excludes the input movie and any already watched movies.
        5. Prevents duplicate sequels from monopolizing recommendations.
        6. Returns top 10 relevant recommendations with mathematically valid similarity scores.
        """
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return []

        if movie_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[movie_id]
        target_cluster = int(self.df.iloc[idx]["cluster"])
        target_vec = self.X_composite[idx:idx+1]
        cluster_name = self.cluster_names[target_cluster]
        target_title = self.df.iloc[idx]["title"]
        target_fkey = get_franchise_key(target_title)

        watched_set = set(int(x) for x in filter_watched_ids if x is not None) if filter_watched_ids else set()
        watched_set.add(movie_id)

        # Candidate selection via K-Means cluster
        cand_indices = [
            i for i, c in enumerate(self.df["cluster"])
            if c == target_cluster and int(self.df.iloc[i]["movie_id"]) not in watched_set
        ]

        # If cluster candidate pool is small (< 30), include closest neighboring centroid
        if len(cand_indices) < 30:
            centroids = self.kmeans.cluster_centers_
            centroid_dists = cosine_similarity(centroids[target_cluster:target_cluster+1], centroids).flatten()
            neighbor_order = np.argsort(centroid_dists)[::-1]
            neighbor_cluster = neighbor_order[1] if len(neighbor_order) > 1 else target_cluster
            extra_indices = [
                i for i, c in enumerate(self.df["cluster"])
                if c == neighbor_cluster and int(self.df.iloc[i]["movie_id"]) not in watched_set
            ]
            cand_indices.extend(extra_indices)

        if not cand_indices:
            cand_indices = [i for i in range(len(self.df)) if int(self.df.iloc[i]["movie_id"]) not in watched_set]

        if not cand_indices:
            return []

        # Genuine cosine similarity on preprocessed multi-modal feature vectors
        cand_matrix = self.X_composite[cand_indices]
        sims = cosine_similarity(target_vec, cand_matrix).flatten()

        cand_df = self.df.iloc[cand_indices].copy()
        cand_df['sim'] = sims

        # Sort strictly by cosine similarity
        sorted_candidates = cand_df.sort_values(by="sim", ascending=False)

        # Two-pass franchise deduplication (maximum 1 movie per franchise in top 10)
        results = []
        seen_franchises = set()
        skipped_for_franchise = []

        for _, row in sorted_candidates.iterrows():
            c_title = row['title']
            c_fkey = get_franchise_key(c_title)

            # Prevent same franchise from occupying multiple recommendation slots
            if c_fkey in seen_franchises:
                skipped_for_franchise.append(row)
                continue

            seen_franchises.add(c_fkey)
            sim_val = float(row['sim'])
            results.append({
                "movie_id": int(row["movie_id"]),
                "title": c_title,
                "year": row.get("release_year") or "",
                "genres": row.get("genres") or "",
                "cluster_id": int(row["cluster"]),
                "cluster_name": self.cluster_names[int(row["cluster"])],
                "score": round(sim_val, 4),
                "cosine_similarity": round(sim_val, 4),
                "rationale": f"Cosine Similarity: {sim_val:.3f} • Cluster #{row['cluster']}: {self.cluster_names[int(row['cluster'])].split('&')[0].strip()}"
            })

            if len(results) >= top_n:
                break

        # Pass 2: Fallback fill to guarantee top_n recommendations if eligible candidates exist
        if len(results) < top_n and skipped_for_franchise:
            for row in skipped_for_franchise:
                sim_val = float(row['sim'])
                results.append({
                    "movie_id": int(row["movie_id"]),
                    "title": row['title'],
                    "year": row.get("release_year") or "",
                    "genres": row.get("genres") or "",
                    "cluster_id": int(row["cluster"]),
                    "cluster_name": self.cluster_names[int(row["cluster"])],
                    "score": round(sim_val, 4),
                    "cosine_similarity": round(sim_val, 4),
                    "rationale": f"Cosine Similarity: {sim_val:.3f} • Cluster #{row['cluster']}: {self.cluster_names[int(row['cluster'])].split('&')[0].strip()} (Franchise Extension)"
                })
                if len(results) >= top_n:
                    break

        return results

    def recommend_by_user_cluster(self, user_interests: dict, top_n: int = 10, filter_watched: bool = True) -> list[dict]:
        """
        User Clustering Recommendation (Option 1):
        1. Rating-Strength Taste Profile:
           - Weight each positive seed movie proportionally to rating strength (e.g. 8.5 vs 7.5).
           - Balances multi-genre seeds (normalizing genre footprints) so single movies don't dominate.
        2. Multi-Cluster Candidate Retrieval:
           - Identifies ALL clusters represented in the user's positive interactions.
           - Draws candidate movies across all active taste clusters + peer cohort consensus.
        3. Multi-Aspect Ranking:
           - Content similarity combines profile affinity with best matching seed affinity.
           - Peer cohort consensus integrates similar users' positive interactions.
           - Sorts strictly by final combined ranking score.
        4. Recommendation Diversity & Deduplication:
           - Prevents a single narrow genre or cluster from monopolizing all 10 recommendations.
           - Strict franchise deduplication (including already watched/rated titles).
        5. Explainable Recommendations:
           - Generates transparent, grounded rationales detailing exact seed movie affinities,
             shared genres, and peer cohort support.
        6. Limited-History Awareness:
           - Detects early-stage profiles (<= 2 ratings) and adapts weights to ground recommendations
             firmly in the user's explicit choices.
        """
        user_pos_ids = [int(x) for x in user_interests.get("positive_movie_ids", []) if x is not None]
        watched_ids = user_interests.get("watched_movie_ids", [])
        user_watched = set(int(x) for x in watched_ids if x is not None)
        user_watched.update(user_pos_ids)
        ratings_dict = user_interests.get("ratings", {})

        # Cold Start / Insufficient History Handler
        if not user_pos_ids:
            return self._get_diverse_exemplars(exclude_ids=user_watched, top_n=top_n)

        pos_indices = [self.id_to_idx[mid] for mid in user_pos_ids if mid in self.id_to_idx]
        if not pos_indices:
            return self._get_diverse_exemplars(exclude_ids=user_watched, top_n=top_n)

        # -------------------------------------------------------------
        # 1. Rating-Strength User Taste Representation
        # -------------------------------------------------------------
        # Compute rating weights: movies rated 8.5 contribute more than movies rated 7.5
        raw_weights = []
        seed_info = []
        for mid in user_pos_ids:
            if mid not in self.id_to_idx:
                continue
            idx = self.id_to_idx[mid]
            row = self.df.iloc[idx]
            r = float(ratings_dict.get(mid, 8.0))
            # Rating strength scaling above 6.5
            w = max(0.5, r - 6.5)
            raw_weights.append(w)
            seed_info.append({
                'mid': mid,
                'idx': idx,
                'title': row['title'],
                'cluster': int(row['cluster']),
                'cluster_name': self.cluster_names[int(row['cluster'])],
                'genres': [g.strip().lower() for g in str(row['genres']).split(',') if g.strip()],
                'rating': r,
                'weight': w,
                'vec': self.X_composite[idx:idx+1]
            })

        if not seed_info:
            return self._get_diverse_exemplars(exclude_ids=user_watched, top_n=top_n)

        total_weight = sum(raw_weights)
        norm_weights = np.array([w / total_weight for w in raw_weights])
        for idx_s, s in enumerate(seed_info):
            s['norm_weight'] = norm_weights[idx_s]

        # Multi-seed user profile vector (weighted by rating strength)
        X_seeds = np.vstack([s['vec'] for s in seed_info])
        u_vec = normalize((X_seeds * norm_weights[:, None]).sum(axis=0).reshape(1, -1), norm='l2', axis=1)

        # Active item clusters in user's history
        active_clusters = list(dict.fromkeys([s['cluster'] for s in seed_info]))
        pred_item_cluster = int(self.kmeans.predict(u_vec)[0])
        if pred_item_cluster not in active_clusters:
            active_clusters.append(pred_item_cluster)

        # -------------------------------------------------------------
        # 2. User Cohort Assignment (with Per-Movie Genre Footprint)
        # -------------------------------------------------------------
        user_cohort_id = None
        user_cohort_name = None
        if self.user_kmeans is not None:
            # Construct genre preference vector: distribute each movie's weight equally across its genres
            u_g_vec = np.zeros((1, len(self.genre_classes_)))
            genre_to_idx = {g: i for i, g in enumerate(self.genre_classes_)}
            for s in seed_info:
                g_list = s['genres']
                if not g_list:
                    continue
                g_share = s['norm_weight'] / len(g_list)
                for g in g_list:
                    if g in genre_to_idx:
                        u_g_vec[0, genre_to_idx[g]] += g_share
            u_g_norm = normalize(u_g_vec, norm='l2', axis=1)

            # Item cluster preference vector
            u_c_vec = np.zeros((1, self.n_clusters))
            for s in seed_info:
                u_c_vec[0, s['cluster']] += s['norm_weight']
            u_c_norm = normalize(u_c_vec, norm='l2', axis=1)

            u_user_feat = normalize(np.hstack([0.60 * u_g_norm, 0.40 * u_c_norm]), norm='l2', axis=1)
            user_cohort_id = int(self.user_kmeans.predict(u_user_feat)[0])
            user_cohort_name = self.user_cluster_names.get(user_cohort_id, f"Cohort {user_cohort_id}")

        # -------------------------------------------------------------
        # 3. Multi-Cluster & Peer Candidate Identification
        # -------------------------------------------------------------
        cohort_peer_scores = self.cohort_peer_ratings.get(user_cohort_id, {})
        all_cand_mids = set()

        # Gather candidates from ALL active clusters represented in user seeds
        for c in active_clusters:
            c_indices = [
                i for i, cl in enumerate(self.df["cluster"])
                if cl == c and int(self.df.iloc[i]["movie_id"]) not in user_watched
            ]
            if not c_indices:
                continue
            c_mat = self.X_composite[c_indices]
            sims = cosine_similarity(u_vec, c_mat).flatten()
            top_k_cluster = np.argsort(sims)[::-1][:60]
            for idx_c in top_k_cluster:
                all_cand_mids.add(int(self.df.iloc[c_indices[idx_c]]["movie_id"]))

        # Also add top peer cohort consensus candidates
        for mid in sorted(cohort_peer_scores, key=cohort_peer_scores.get, reverse=True)[:60]:
            if mid in self.id_to_idx and mid not in user_watched:
                all_cand_mids.add(mid)

        if not all_cand_mids:
            all_cand_mids = set(int(self.df.iloc[i]["movie_id"]) for i in range(len(self.df)) if int(self.df.iloc[i]["movie_id"]) not in user_watched)

        cand_mids_list = [mid for mid in all_cand_mids if mid in self.id_to_idx]
        if not cand_mids_list:
            return []

        cand_indices_list = [self.id_to_idx[mid] for mid in cand_mids_list]
        cand_matrix = self.X_composite[cand_indices_list]

        # -------------------------------------------------------------
        # 4. Multi-Aspect Similarity & Ranking
        # -------------------------------------------------------------
        # 1. Similarity to overall user taste profile
        prof_sims = cosine_similarity(u_vec, cand_matrix).flatten()

        # 2. Specific affinity to individual positive seeds
        seed_sims_matrix = cosine_similarity(X_seeds, cand_matrix) # shape: (n_seeds, n_cands)
        best_seed_indices = np.argmax(seed_sims_matrix, axis=0)
        best_seed_sims = np.max(seed_sims_matrix, axis=0)

        # 3. Content similarity: blends overall profile fit with specific seed affinity
        content_sims = 0.50 * prof_sims + 0.50 * best_seed_sims
        peer_scores = np.array([float(cohort_peer_scores.get(mid, 0.0)) for mid in cand_mids_list])

        # Adaptive weighting based on history depth:
        # Limited history (<= 2 movies): content similarity carries 75% weight
        # Established history (>= 3 movies): 70% content + 30% peer consensus
        is_limited_history = len(user_pos_ids) <= 2
        w_content = 0.75 if is_limited_history else 0.70
        w_peer = 0.25 if is_limited_history else 0.30
        final_scores = w_content * content_sims + w_peer * peer_scores

        cand_df = self.df.iloc[cand_indices_list].copy()
        cand_df['content_sim'] = content_sims
        cand_df['peer_score'] = peer_scores
        cand_df['final_score'] = final_scores
        cand_df['best_seed_idx'] = best_seed_indices
        cand_df['best_seed_sim'] = best_seed_sims

        sorted_candidates = cand_df.sort_values(by="final_score", ascending=False)

        # -------------------------------------------------------------
        # 5. Diversity Balancing & Franchise Deduplication
        # -------------------------------------------------------------
        # Exclude franchises already watched or rated by the user
        watched_fkeys = set()
        for wid in user_watched:
            if wid in self.id_to_idx:
                watched_fkeys.add(get_franchise_key(self.df.iloc[self.id_to_idx[wid]]['title']))

        seen_franchises = set(watched_fkeys)
        cluster_counts = Counter()
        selected_rows = []
        skipped_for_cluster_cap = []

        # Target representation: across multiple active clusters, prevent single-cluster domination (max 4 per cluster)
        max_per_cluster = 4 if len(active_clusters) >= 2 else top_n

        # Pre-pass: If the user demonstrated interest in multiple distinct seed clusters,
        # ensure each active seed cluster gets at least 2 top candidates if eligible candidates exist!
        seed_clusters = list(dict.fromkeys([s['cluster'] for s in seed_info]))
        if len(seed_clusters) >= 2:
            target_seed_mins = {c: 2 for c in seed_clusters}
            for c, min_cnt in target_seed_mins.items():
                c_pool = sorted_candidates[sorted_candidates['cluster'] == c]
                added = 0
                for _, row in c_pool.iterrows():
                    fkey = get_franchise_key(row['title'])
                    if fkey in seen_franchises:
                        continue
                    seen_franchises.add(fkey)
                    cluster_counts[c] += 1
                    selected_rows.append(row)
                    added += 1
                    if added >= min_cnt:
                        break

        # Pass 1: Add remaining open slots respecting franchise uniqueness and cluster diversity cap
        for _, row in sorted_candidates.iterrows():
            if len(selected_rows) >= top_n:
                break
            c_title = row['title']
            c_fkey = get_franchise_key(c_title)
            c_cluster = int(row['cluster'])

            if c_fkey in seen_franchises:
                continue

            if cluster_counts[c_cluster] >= max_per_cluster:
                skipped_for_cluster_cap.append(row)
                continue

            seen_franchises.add(c_fkey)
            cluster_counts[c_cluster] += 1
            selected_rows.append(row)

        # Pass 2: Fallback fill if needed
        if len(selected_rows) < top_n and skipped_for_cluster_cap:
            for row in skipped_for_cluster_cap:
                c_fkey = get_franchise_key(row['title'])
                if c_fkey in seen_franchises:
                    continue
                seen_franchises.add(c_fkey)
                selected_rows.append(row)
                if len(selected_rows) >= top_n:
                    break

        # Final guarantee: sort selected recommendations strictly by final_score descending
        selected_rows.sort(key=lambda r: float(r['final_score']), reverse=True)

        # -------------------------------------------------------------
        # 6. Detailed, Model-Grounded Explanations
        # -------------------------------------------------------------
        results = []
        for row in selected_rows:
            c_title = row['title']
            c_cluster = int(row['cluster'])
            c_cluster_name = self.cluster_names[c_cluster].split('&')[0].strip()
            final_sc = float(row['final_score'])
            csim = float(row['content_sim'])
            pscore = float(row['peer_score'])
            best_s = seed_info[int(row['best_seed_idx'])]
            best_s_title = best_s['title']
            best_s_sim = float(row['best_seed_sim'])

            # Shared genres between candidate and matched seed
            cand_genres = set(g.strip().lower() for g in str(row.get('genres') or '').split(',') if g.strip())
            shared_genres = [g.title() for g in list(cand_genres.intersection(set(best_s['genres'])))[:2]]
            shared_str = f" ({', '.join(shared_genres)})" if shared_genres else ""

            # Determine specific basis of match
            if best_s_sim >= 0.40 and pscore >= 0.50:
                explanation = f"High affinity to your rated '{best_s_title}'{shared_str} & strong consensus from your {user_cohort_name}"
            elif best_s_sim >= 0.40:
                explanation = f"High affinity to your rated movie '{best_s_title}'{shared_str} • Cluster #{c_cluster}: {c_cluster_name}"
            elif pscore >= 0.60:
                explanation = f"Top-rated by users with similar taste in your {user_cohort_name} • Cluster #{c_cluster}: {c_cluster_name}"
            elif shared_genres:
                explanation = f"Shares preferred {', '.join(shared_genres)} themes • Cluster #{c_cluster}: {c_cluster_name}"
            else:
                explanation = f"Thematic alignment in Cluster #{c_cluster}: {c_cluster_name}"

            results.append({
                "movie_id": int(row["movie_id"]),
                "title": c_title,
                "year": row.get("release_year") or "",
                "genres": row.get("genres") or "",
                "cluster_id": c_cluster,
                "cluster_name": self.cluster_names[c_cluster],
                "user_cohort_id": user_cohort_id,
                "user_cohort_name": user_cohort_name,
                "score": round(final_sc, 4),
                "final_score": round(final_sc, 4),
                "content_similarity": round(csim, 4),
                "peer_support": round(pscore, 4),
                "matched_seed_title": best_s_title,
                "matched_seed_similarity": round(best_s_sim, 4),
                "is_limited_history": is_limited_history,
                "rationale": f"Rank Score: {final_sc:.3f} (Content Sim: {csim:.3f}, Peer Support: {pscore:.3f}) • {explanation}"
            })

        return results

    def _get_diverse_exemplars(self, exclude_ids: set, top_n: int = 10) -> list[dict]:
        """Provides balanced top-rated exemplars across diverse clusters for cold-start users."""
        results = []
        for c in range(self.n_clusters):
            c_df = self.df[(self.df['cluster'] == c) & (~self.df['movie_id'].isin(exclude_ids))].sort_values(
                by=["final_score", "pop"], ascending=[False, False]
            )
            if not c_df.empty:
                row = c_df.iloc[0]
                results.append({
                    "movie_id": int(row["movie_id"]),
                    "title": row["title"],
                    "year": row.get("release_year") or "",
                    "genres": row.get("genres") or "",
                    "cluster_id": c,
                    "cluster_name": self.cluster_names[c],
                    "score": round(float(row.get("final_score", 0.8)), 4),
                    "rationale": f"Cluster #{c} Exemplar: {self.cluster_names[c]}"
                })
            if len(results) >= top_n:
                break
        return results

    def get_cluster_metrics(self) -> dict:
        """Returns mathematical validation metrics for unsupervised evaluation."""
        return {
            "n_clusters": self.n_clusters,
            "n_item_clusters": self.n_clusters,
            "inertia": self.inertia,
            "item_inertia": self.inertia,
            "silhouette_score": self.silhouette,
            "item_silhouette_score": self.silhouette,
            "n_user_clusters": self.n_user_clusters,
            "user_inertia": self.user_inertia,
            "user_silhouette_score": self.user_silhouette,
            "pca_explained_variance_ratio": self.pca_evr,
            "user_pca_explained_variance_ratio": self.user_pca_evr,
            "total_movies": len(self.df),
            "total_users": len(self.user_cohort_assignments)
        }

    def generate_svg_visualization(self, output_path: str = "reports/movie_clusters_pca.svg") -> str:
        """Generates a standalone, crisp SVG vector graphic of the 2D PCA cluster space."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        w, h = 1000, 700
        pad = 60

        min_x, max_x = self.df['pca_x'].min(), self.df['pca_x'].max()
        min_y, max_y = self.df['pca_y'].min(), self.df['pca_y'].max()

        def map_x(val):
            return pad + (val - min_x) / (max_x - min_x) * (w - 2 * pad)

        def map_y(val):
            return h - pad - (val - min_y) / (max_y - min_y) * (h - 2 * pad)

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            f'<rect width="{w}" height="{h}" fill="#0f172a"/>',
            f'<text x="{w/2}" y="32" font-family="-apple-system, sans-serif" font-size="18" font-weight="bold" fill="#38bdf8" text-anchor="middle">Movie Clusters in 2D PCA Latent Space (K-Means K={self.n_clusters})</text>',
            f'<text x="{w/2}" y="50" font-family="-apple-system, sans-serif" font-size="11" fill="#94a3b8" text-anchor="middle">Principal Component 1 ({self.pca_evr[0]*100:.1f}%) vs Principal Component 2 ({self.pca_evr[1]*100:.1f}%)</text>'
        ]

        # Draw movie points
        for _, row in self.df.iterrows():
            sx = round(map_x(row['pca_x']), 1)
            sy = round(map_y(row['pca_y']), 1)
            color = self.cluster_colors[row['cluster']]
            svg.append(f'<circle cx="{sx}" cy="{sy}" r="2" fill="{color}" opacity="0.6"/>')

        # Draw cluster centroid labels
        for c in range(self.n_clusters):
            c_df = self.df[self.df['cluster'] == c]
            if c_df.empty:
                continue
            cx = map_x(c_df['pca_x'].mean())
            cy = map_y(c_df['pca_y'].mean())
            color = self.cluster_colors[c]
            name = self.cluster_names[c]
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
            svg.append(f'<text x="{cx:.1f}" y="{cy-12:.1f}" font-family="-apple-system, sans-serif" font-size="10" font-weight="600" fill="{color}" text-anchor="middle">{name}</text>')

        # Legend
        svg.append(f'<g transform="translate(60, {h-35})">')
        for c in range(self.n_clusters):
            col_x = c * 90
            color = self.cluster_colors[c]
            name = self.cluster_names[c].split('&')[0].strip()
            svg.append(f'<rect x="{col_x}" y="0" width="8" height="8" fill="{color}" rx="2"/>')
            svg.append(f'<text x="{col_x+12}" y="8" font-family="-apple-system, sans-serif" font-size="8" fill="#cbd5e1">#{c} {name}</text>')
        svg.append('</g>')

        svg.append('</svg>')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(svg))
        return os.path.abspath(output_path)

    def generate_html_visualization(self, output_path: str = "reports/movie_clusters_pca.html", highlight_movie_id: int = None) -> str:
        """
        Generates an interactive HTML5 PCA dashboard featuring:
        - Dual View Toggle: Movie Clusters (5,915 Items) vs User Taste Cohorts (612 Users).
        - Genuine PCA 2D coordinates for both movies and users.
        - Search and Highlighting for any movie or user.
        - Detailed hover cards, pan, zoom, reset, and cluster filter checkboxes.
        - Real-time PCA Explained Variance Ratios for both models.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        movie_points = []
        for _, row in self.df.iterrows():
            movie_points.append({
                "id": int(row["movie_id"]),
                "t": row["title"],
                "y": int(row["release_year"]) if pd.notnull(row["release_year"]) else 0,
                "g": str(row["genres"] or ""),
                "r": float(row["rating"]) if pd.notnull(row["rating"]) else 0.0,
                "c": int(row["cluster"]),
                "px": round(float(row["pca_x"]), 4),
                "py": round(float(row["pca_y"]), 4)
            })

        movie_cluster_meta = []
        for c in range(self.n_clusters):
            c_df = self.df[self.df["cluster"] == c]
            movie_cluster_meta.append({
                "id": c,
                "name": self.cluster_names[c],
                "color": self.cluster_colors[c],
                "count": len(c_df),
                "cx": round(float(c_df["pca_x"].mean()), 4) if not c_df.empty else 0.0,
                "cy": round(float(c_df["pca_y"].mean()), 4) if not c_df.empty else 0.0
            })

        user_cluster_meta = []
        for c in range(self.n_user_clusters):
            u_in_c = [u for u in self.user_points if u['c'] == c]
            user_cluster_meta.append({
                "id": c,
                "name": self.user_cluster_names.get(c, f"Cohort {c}"),
                "color": self.USER_PALETTE[c % len(self.USER_PALETTE)],
                "count": len(u_in_c),
                "cx": round(float(np.mean([u['px'] for u in u_in_c])), 4) if u_in_c else 0.0,
                "cy": round(float(np.mean([u['py'] for u in u_in_c])), 4) if u_in_c else 0.0
            })

        movies_json = json.dumps(movie_points)
        movie_meta_json = json.dumps(movie_cluster_meta)
        users_json = json.dumps(self.user_points)
        user_meta_json = json.dumps(user_cluster_meta)

        item_evr_str = f"PC1: {self.pca_evr[0]*100:.1f}%, PC2: {self.pca_evr[1]*100:.1f}% (Total: {sum(self.pca_evr)*100:.1f}%)"
        user_evr_str = f"PC1: {self.user_pca_evr[0]*100:.1f}%, PC2: {self.user_pca_evr[1]*100:.1f}% (Total: {sum(self.user_pca_evr)*100:.1f}%)" if self.user_pca_evr else "N/A"
        initial_hl_id = highlight_movie_id if highlight_movie_id else "null"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MovieDude - Interactive PCA & K-Means Cluster Visualizer</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #090d16;
      color: #f1f5f9;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }}
    #sidebar {{
      width: 380px;
      background: #0f172a;
      border-right: 1px solid #1e293b;
      display: flex;
      flex-direction: column;
      padding: 20px;
      gap: 16px;
      z-index: 10;
      overflow-y: auto;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .brand h1 {{
      font-size: 19px;
      font-weight: 800;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .view-toggle {{
      display: flex;
      background: #1e293b;
      border-radius: 8px;
      padding: 4px;
      gap: 4px;
    }}
    .view-btn {{
      flex: 1;
      padding: 8px 12px;
      border: none;
      background: transparent;
      color: #94a3b8;
      font-weight: 600;
      font-size: 13px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .view-btn.active {{
      background: #0284c7;
      color: #ffffff;
      box-shadow: 0 2px 6px rgba(2, 132, 199, 0.4);
    }}
    .stats-card {{
      background: #1e293b;
      border-radius: 10px;
      padding: 14px;
      border: 1px solid #334155;
      font-size: 12px;
      line-height: 1.6;
    }}
    .stats-row {{ display: flex; justify-content: space-between; }}
    .stats-label {{ color: #94a3b8; }}
    .stats-val {{ font-weight: 700; color: #38bdf8; }}
    .search-box input {{
      width: 100%;
      padding: 10px 14px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #f8fafc;
      font-size: 13px;
      outline: none;
    }}
    .search-box input:focus {{ border-color: #38bdf8; }}
    .clusters-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 380px;
      overflow-y: auto;
    }}
    .cluster-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: #131c31;
      border-radius: 6px;
      cursor: pointer;
      font-size: 12px;
      transition: background 0.15s;
    }}
    .cluster-item:hover {{ background: #1e293b; }}
    .cluster-dot {{ width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }}
    .cluster-title {{ flex: 1; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .cluster-count {{ color: #64748b; font-size: 11px; }}
    #viewport {{
      flex: 1;
      position: relative;
      background: radial-gradient(circle at 50% 50%, #0d1527 0%, #060911 100%);
      cursor: grab;
    }}
    #viewport:active {{ cursor: grabbing; }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    .controls {{
      position: absolute;
      top: 20px;
      right: 20px;
      display: flex;
      gap: 8px;
      background: rgba(15, 23, 42, 0.85);
      backdrop-filter: blur(8px);
      padding: 6px;
      border-radius: 8px;
      border: 1px solid #334155;
    }}
    .ctrl-btn {{
      background: #1e293b;
      border: none;
      color: #f1f5f9;
      width: 32px;
      height: 32px;
      border-radius: 6px;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .ctrl-btn:hover {{ background: #334155; }}
    #tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(15, 23, 42, 0.95);
      border: 1px solid #38bdf8;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 12px;
      color: #f8fafc;
      box-shadow: 0 10px 25px rgba(0,0,0,0.6);
      transform: translate(-50%, -120%);
      display: none;
      min-width: 220px;
      z-index: 100;
    }}
    #tooltip h3 {{ font-size: 14px; margin-bottom: 4px; color: #38bdf8; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="brand">
      <h1>MovieDude Visualizer</h1>
    </div>

    <div class="view-toggle">
      <button id="btnMovieView" class="view-btn active" onclick="switchView('movies')">🎬 Movie Items</button>
      <button id="btnUserView" class="view-btn" onclick="switchView('users')">👥 User Cohorts</button>
    </div>

    <div class="stats-card">
      <div class="stats-row"><span class="stats-label">Clustering Algorithm:</span><span class="stats-val">K-Means</span></div>
      <div class="stats-row"><span class="stats-label">Dimensionality Reduction:</span><span class="stats-val">2D PCA</span></div>
      <div class="stats-row"><span class="stats-label" id="lblClusters">Item Clusters (K):</span><span class="stats-val" id="valClusters">{self.n_clusters}</span></div>
      <div class="stats-row"><span class="stats-label" id="lblTotal">Total Data Points:</span><span class="stats-val" id="valTotal">{len(self.df)}</span></div>
      <div class="stats-row"><span class="stats-label">PCA Explained Variance:</span><span class="stats-val" id="valEvr">{item_evr_str}</span></div>
    </div>

    <div class="search-box">
      <input type="text" id="searchInput" placeholder="Search by title (e.g. La La Land, Whiplash)..." oninput="onSearch(this.value)">
    </div>

    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700;">Filter Clusters</div>
    <div class="clusters-list" id="clustersList"></div>
  </div>

  <div id="viewport">
    <canvas id="scatterCanvas"></canvas>
    <div class="controls">
      <button class="ctrl-btn" onclick="zoom(1.25)" title="Zoom In">+</button>
      <button class="ctrl-btn" onclick="zoom(0.8)" title="Zoom Out">−</button>
      <button class="ctrl-btn" onclick="resetView()" title="Reset View">⟲</button>
    </div>
    <div id="tooltip">
      <h3 id="ttTitle">Title</h3>
      <div id="ttMeta" style="color: #94a3b8; margin-bottom: 6px;">Year • Genres</div>
      <div id="ttCluster" style="font-weight: 600;">Cluster</div>
      <div id="ttCoords" style="color: #64748b; font-size: 10px; margin-top: 4px;">PCA: (0.0, 0.0)</div>
    </div>
  </div>

  <script>
    const moviePoints = {movies_json};
    const movieClusters = {movie_meta_json};
    const userPoints = {users_json};
    const userClusters = {user_meta_json};
    const initialHlId = {initial_hl_id};

    let currentMode = 'movies'; // 'movies' or 'users'
    let activeClusters = new Set();
    let highlightedPoint = null;
    let hoveredPoint = null;
    let searchQuery = '';

    const canvas = document.getElementById('scatterCanvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');

    let camera = {{ x: 0, y: 0, zoom: 1 }};
    let isDragging = false;
    let dragStart = {{ x: 0, y: 0 }};

    function switchView(mode) {{
      currentMode = mode;
      document.getElementById('btnMovieView').className = 'view-btn' + (mode === 'movies' ? ' active' : '');
      document.getElementById('btnUserView').className = 'view-btn' + (mode === 'users' ? ' active' : '');

      const isM = (mode === 'movies');
      document.getElementById('lblClusters').innerText = isM ? 'Item Clusters (K):' : 'User Cohorts (K):';
      document.getElementById('valClusters').innerText = isM ? movieClusters.length : userClusters.length;
      document.getElementById('lblTotal').innerText = isM ? 'Total Movies:' : 'Total Users:';
      document.getElementById('valTotal').innerText = isM ? moviePoints.length : userPoints.length;
      document.getElementById('valEvr').innerText = isM ? '{item_evr_str}' : '{user_evr_str}';
      document.getElementById('searchInput').placeholder = isM ? 'Search by title (e.g. La La Land)...' : 'Search by user ID...';

      highlightedPoint = null;
      renderClustersList();
      resetView();
    }}

    function renderClustersList() {{
      const list = document.getElementById('clustersList');
      list.innerHTML = '';
      const clusters = (currentMode === 'movies') ? movieClusters : userClusters;
      activeClusters = new Set(clusters.map(c => c.id));

      clusters.forEach(c => {{
        const item = document.createElement('div');
        item.className = 'cluster-item';
        item.innerHTML = `
          <div class="cluster-dot" style="background: ${{c.color}}"></div>
          <div class="cluster-title" title="${{c.name}}">#${{c.id}} ${{c.name}}</div>
          <div class="cluster-count">${{c.count}}</div>
        `;
        item.onclick = () => {{
          if (activeClusters.has(c.id) && activeClusters.size === 1) {{
            activeClusters = new Set(clusters.map(x => x.id));
          }} else {{
            activeClusters.clear();
            activeClusters.add(c.id);
          }}
          draw();
        }};
        list.appendChild(item);
      }});
    }}

    function resize() {{
      canvas.width = canvas.parentElement.clientWidth;
      canvas.height = canvas.parentElement.clientHeight;
      draw();
    }}
    window.addEventListener('resize', resize);

    function resetView() {{
      const points = (currentMode === 'movies') ? moviePoints : userPoints;
      if (!points.length) return;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      points.forEach(p => {{
        if (p.px < minX) minX = p.px;
        if (p.px > maxX) maxX = p.px;
        if (p.py < minY) minY = p.py;
        if (p.py > maxY) maxY = p.py;
      }});
      const rangeX = maxX - minX || 1;
      const rangeY = maxY - minY || 1;
      const scaleX = (canvas.width - 160) / rangeX;
      const scaleY = (canvas.height - 160) / rangeY;
      camera.zoom = Math.min(scaleX, scaleY);
      camera.x = canvas.width / 2 - ((minX + maxX) / 2) * camera.zoom;
      camera.y = canvas.height / 2 - ((minY + maxY) / 2) * camera.zoom;
      draw();
    }}

    function toScreen(px, py) {{
      return {{
        x: camera.x + px * camera.zoom,
        y: camera.y + py * camera.zoom
      }};
    }}

    function toWorld(sx, sy) {{
      return {{
        px: (sx - camera.x) / camera.zoom,
        py: (sy - camera.y) / camera.zoom
      }};
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Grid Lines
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;
      const origin = toScreen(0, 0);
      ctx.beginPath();
      ctx.moveTo(0, origin.y); ctx.lineTo(canvas.width, origin.y);
      ctx.moveTo(origin.x, 0); ctx.lineTo(origin.x, canvas.height);
      ctx.stroke();

      const points = (currentMode === 'movies') ? moviePoints : userPoints;
      const clusters = (currentMode === 'movies') ? movieClusters : userClusters;
      const colorMap = Object.fromEntries(clusters.map(c => [c.id, c.color]));

      // Draw normal points
      points.forEach(p => {{
        if (!activeClusters.has(p.c)) return;
        const s = toScreen(p.px, p.py);
        if (s.x < -10 || s.x > canvas.width + 10 || s.y < -10 || s.y > canvas.height + 10) return;

        const isMatch = searchQuery && (
          (currentMode === 'movies' && p.t.toLowerCase().includes(searchQuery)) ||
          (currentMode === 'users' && String(p.id).toLowerCase().includes(searchQuery))
        );

        ctx.fillStyle = colorMap[p.c] || '#38bdf8';
        ctx.globalAlpha = isMatch ? 1.0 : (searchQuery ? 0.2 : 0.65);
        ctx.beginPath();
        const r = isMatch ? 6 : (currentMode === 'movies' ? 3 : 4);
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.fill();

        if (isMatch) {{
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }}
      }});

      // Draw Centroids
      ctx.globalAlpha = 1.0;
      clusters.forEach(c => {{
        if (!activeClusters.has(c.id)) return;
        const s = toScreen(c.cx, c.cy);
        ctx.fillStyle = c.color;
        ctx.beginPath();
        ctx.arc(s.x, s.y, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.font = 'bold 11px -apple-system, sans-serif';
        ctx.fillStyle = '#f8fafc';
        ctx.textAlign = 'center';
        ctx.fillText(c.name.split('&')[0].trim(), s.x, s.y - 12);
      }});

      // Highlight selected point
      if (highlightedPoint) {{
        const s = toScreen(highlightedPoint.px, highlightedPoint.py);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(s.x, s.y, 12, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(s.x, s.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    function onSearch(val) {{
      searchQuery = val.trim().toLowerCase();
      if (searchQuery) {{
        const points = (currentMode === 'movies') ? moviePoints : userPoints;
        const match = points.find(p => (currentMode === 'movies' ? p.t.toLowerCase().includes(searchQuery) : String(p.id).toLowerCase().includes(searchQuery)));
        if (match) {{
          highlightedPoint = match;
        }}
      }}
      draw();
    }}

    function zoom(factor) {{
      camera.zoom *= factor;
      draw();
    }}

    // Mouse Interactions
    canvas.addEventListener('mousedown', e => {{
      isDragging = true;
      dragStart = {{ x: e.clientX - camera.x, y: e.clientY - camera.y }};
    }});
    window.addEventListener('mouseup', () => isDragging = false);
    canvas.addEventListener('mousemove', e => {{
      if (isDragging) {{
        camera.x = e.clientX - dragStart.x;
        camera.y = e.clientY - dragStart.y;
        draw();
        return;
      }}

      // Hover Hit Testing
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const points = (currentMode === 'movies') ? moviePoints : userPoints;
      const clusters = (currentMode === 'movies') ? movieClusters : userClusters;

      let found = null;
      for (let i = points.length - 1; i >= 0; i--) {{
        const p = points[i];
        if (!activeClusters.has(p.c)) continue;
        const s = toScreen(p.px, p.py);
        const dist = Math.hypot(mouseX - s.x, mouseY - s.y);
        if (dist <= 8) {{ found = p; break; }}
      }}

      if (found) {{
        hoveredPoint = found;
        const cMeta = clusters.find(c => c.id === found.c) || {{ name: 'Cluster ' + found.c }};
        tooltip.style.display = 'block';
        tooltip.style.left = mouseX + 'px';
        tooltip.style.top = mouseY + 'px';

        if (currentMode === 'movies') {{
          document.getElementById('ttTitle').innerText = found.t;
          document.getElementById('ttMeta').innerText = `${{found.y}} • ${{found.g}} • ★ ${{found.r.toFixed(1)}}`;
          document.getElementById('ttCluster').innerText = `Item Cluster #${{found.c}}: ${{cMeta.name}}`;
        }} else {{
          document.getElementById('ttTitle').innerText = `User: ${{found.id}}`;
          document.getElementById('ttMeta').innerText = `Total Ratings: ${{found.count}} • Top Genre: ${{found.top_genre}}`;
          document.getElementById('ttCluster').innerText = `Taste Cohort #${{found.c}}: ${{cMeta.name}}`;
        }}
        document.getElementById('ttCoords').innerText = `PCA 2D Coordinates: (x: ${{found.px.toFixed(3)}}, y: ${{found.py.toFixed(3)}})`;
      }} else {{
        tooltip.style.display = 'none';
        hoveredPoint = null;
      }}
    }});

    canvas.addEventListener('click', () => {{
      if (hoveredPoint) {{
        highlightedPoint = hoveredPoint;
        draw();
      }}
    }});

    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      zoom(zoomFactor);
    }}, {{ passive: false }});

    // Initialize
    renderClustersList();
    resize();
    resetView();

    if (initialHlId) {{
      const match = moviePoints.find(p => p.id === initialHlId);
      if (match) {{
        highlightedPoint = match;
        draw();
      }}
    }}
  </script>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return os.path.abspath(output_path)

def get_clustering_model(db_path: str) -> MovieClusterModel:
    """Singleton getter for the clustering recommendation and PCA engine."""
    global _GLOBAL_CLUSTER_MODEL
    if _GLOBAL_CLUSTER_MODEL is None:
        _GLOBAL_CLUSTER_MODEL = MovieClusterModel(db_path)
    return _GLOBAL_CLUSTER_MODEL

def invalidate_cluster_cache():
    """Resets the singleton model cache when new ratings are logged."""
    global _GLOBAL_CLUSTER_MODEL
    _GLOBAL_CLUSTER_MODEL = None
