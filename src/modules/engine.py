import re
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modules.watched_movies import catch_watched_movies

STOP_WORDS = {
    'based on novel or book', 'duringcreditsstinger', 'aftercreditsstinger', 
    'film', 'short story', 'independent film', 'in'
}

# High-frequency umbrella genres that otherwise dilute thematic relevancy
UMBRELLA_GENRES = {'drama', 'history', 'comedy', 'romance', 'action'}

# In-memory ratings cache for ultra-low latency response
_RATINGS_CACHE = None
_MOVIE_NORMS_CACHE = None

def invalidate_ratings_cache():
    """Resets the in-memory ratings cache and clustering model when user rates or modifies movie data."""
    global _RATINGS_CACHE, _MOVIE_NORMS_CACHE
    _RATINGS_CACHE = None
    _MOVIE_NORMS_CACHE = None
    try:
        from modules.clustering import invalidate_cluster_cache
        invalidate_cluster_cache()
    except Exception:
        pass

def _slugify_list(text: str) -> list[str]:
    """
    Glues multi-word names and phrases with underscores.
    e.g. 'Nick Cassavetes' -> 'nick_cassavetes'
         'Ryan Gosling' -> 'ryan_gosling'
         'psychological thriller' -> 'psychological_thriller'
    This prevents word-bleed where 'Nick' matches 'Nick of Time'.
    """
    if not text:
        return []
    items = [i.strip() for i in text.split(',') if i.strip()]
    slugged = []
    for item in items:
        s = re.sub(r'[\s\-]+', '_', item.lower().strip())
        if s and s not in STOP_WORDS:
            slugged.append(s)
    return slugged

def _format_weighted_soup(director_tokens, stars_tokens, genres_tokens, keywords_tokens, studio_tokens, country_tokens) -> str:
    """
    Constructs the feature soup using a hierarchical weighting scheme:
    - High Weight (w = 0.5): Niche sub-themes & keywords (repeated 3x)
    - Medium Weight (w = 0.3): Primary/niche genres (repeated 2x) (e.g. Sports, Biography, Sci-Fi)
    - Low Weight (w = 0.2): Broad umbrella genres (1x unrepeated) (e.g. Drama, History, Comedy)
    - Identity Metadata: Studio (2x), Director (1x), Stars (1x), Country (1x)
    """
    # 1. High Weight: Sub-themes & plot keywords
    keywords_part = f"{' '.join(keywords_tokens)} {' '.join(keywords_tokens)} {' '.join(keywords_tokens)}" if keywords_tokens else ""
    
    # 2. Medium Weight: Niche / Primary genres (Sports, Biography, Animation, War, Music, etc.)
    niche_genres = [g for g in genres_tokens if g not in UMBRELLA_GENRES]
    umbrella_genres = [g for g in genres_tokens if g in UMBRELLA_GENRES]
    niche_part = f"{' '.join(niche_genres)} {' '.join(niche_genres)}" if niche_genres else ""
    
    # 3. Low Weight: High-frequency umbrella genres (Drama, History, Romance)
    umbrella_part = " ".join(umbrella_genres)
    
    # 4. Identity tokens
    director_part = " ".join(director_tokens)
    stars_part = " ".join(stars_tokens)
    studio_part = f"{' '.join(studio_tokens)} {' '.join(studio_tokens)}" if studio_tokens else ""
    country_part = " ".join(country_tokens) if country_tokens else ""

    return f"{keywords_part} {niche_part} {umbrella_part} {director_part} {stars_part} {studio_part} {country_part}".strip()

def _build_soup_and_density(row) -> tuple[str, int]:
    """Builds the metadata soup without duplicating the title."""
    director_tokens = _slugify_list(row.get('director') or '')
    stars_tokens = _slugify_list(row.get('stars') or '')
    genres_tokens = _slugify_list(row.get('genres') or '')
    keywords_tokens = _slugify_list(row.get('keywords') or '')
    studio_tokens = _slugify_list(row.get('studio') or '')
    country_tokens = _slugify_list(row.get('country') or '')

    kw_count = len(keywords_tokens)
    soup = _format_weighted_soup(director_tokens, stars_tokens, genres_tokens, keywords_tokens, studio_tokens, country_tokens)
    return soup, kw_count

def _franchise_boost(target_title: str, candidate_title: str) -> float:
    """
    Controlled sequel boost: awards a modest bonus (+0.12) only if candidate shares
    the same franchise root name (e.g. 'Toy Story' and 'Toy Story 2', 'Drishyam' and 'Drishyam 2').
    """
    if not target_title or not candidate_title:
        return 0.0
    t1 = re.sub(r'[:\-\d]+.*$', '', target_title).strip().lower()
    t2 = re.sub(r'[:\-\d]+.*$', '', candidate_title).strip().lower()
    if len(t1) >= 4 and t1 == t2 and target_title.lower() != candidate_title.lower():
        return 0.12
    return 0.0

def _get_cached_ratings(db_path: str):
    global _RATINGS_CACHE, _MOVIE_NORMS_CACHE
    if _RATINGS_CACHE is None:
        try:
            with sqlite3.connect(db_path) as conn:
                _RATINGS_CACHE = pd.read_sql("SELECT user_id, movie_id, user_rate FROM Users_data", conn)
            _MOVIE_NORMS_CACHE = _RATINGS_CACHE.groupby('movie_id')['user_rate'].apply(
                lambda x: np.sqrt(np.sum(x**2))
            ).to_dict()
        except Exception as e:
            print(f"[Warning] Could not load ratings cache: {e}")
            _RATINGS_CACHE = pd.DataFrame()
            _MOVIE_NORMS_CACHE = {}
    return _RATINGS_CACHE, _MOVIE_NORMS_CACHE

def fast_item_cf(db_path: str, target_movie_id: int, min_co_raters: int = 3, shrinkage_k: float = 10.0) -> dict[int, float]:
    """
    Computes Item-Item Collaborative Filtering scores with significance shrinkage.
    Rejects high-variance spurious pairs with fewer than min_co_raters common raters.
    Runs in < 50ms using subset-vectorized dot products.
    """
    if not target_movie_id:
        return {}
    df_ratings, movie_norms = _get_cached_ratings(db_path)
    if df_ratings.empty:
        return {}

    target_ratings = df_ratings[df_ratings['movie_id'] == target_movie_id]
    if len(target_ratings) < min_co_raters:
        return {}
    
    target_norm = movie_norms.get(target_movie_id, 0.0)
    if target_norm == 0.0:
        return {}

    target_user_dict = dict(zip(target_ratings['user_id'], target_ratings['user_rate']))
    co_ratings = df_ratings[df_ratings['user_id'].isin(target_user_dict.keys())].copy()
    co_ratings['target_rate'] = co_ratings['user_id'].map(target_user_dict)
    co_ratings['prod'] = co_ratings['user_rate'] * co_ratings['target_rate']

    agg = co_ratings.groupby('movie_id').agg(
        dot=('prod', 'sum'),
        co_count=('user_id', 'count')
    )
    agg = agg[agg.index != target_movie_id]
    agg = agg[agg['co_count'] >= min_co_raters]
    if agg.empty:
        return {}

    scores = {}
    for mid, row in agg.iterrows():
        cand_norm = movie_norms.get(mid, 0.0)
        if cand_norm > 0:
            raw_sim = row['dot'] / (target_norm * cand_norm)
            shrinkage = min(row['co_count'] / shrinkage_k, 1.0)
            scores[mid] = float(raw_sim * shrinkage)
    return scores

def _compute_rationale(rec_input, cand_row, cf_score: float = 0.0) -> str:
    """Generates a human-interpretable rationale for why a movie was recommended."""
    # CASE 1: Personalized User Feed (Option 1) -> "Because you loved '[Movie]'"
    if isinstance(rec_input, dict) and "pos_movies_dict" in rec_input and rec_input["pos_movies_dict"]:
        pos_dict = rec_input["pos_movies_dict"]
        cand_k = set(k.lower().strip() for k in (cand_row.get('keywords') or '').split(',') if k.strip())
        cand_g = set(g.lower().strip() for g in (cand_row.get('genres') or '').split(',') if g.strip())
        cand_dir = (cand_row.get('director') or '').lower().strip()
        cand_studio = (cand_row.get('studio') or '').lower().strip()

        best_seed = None
        best_overlap = 0
        best_match_tag = ''

        cand_title = cand_row.get("title", "")
        for mid, s_data in pos_dict.items():
            overlap = 0
            tag = ''
            s_title = s_data.get('title', '')
            if s_title and cand_title:
                t1 = re.sub(r'[:\-\d]+.*$', '', s_title).strip().lower()
                t2 = re.sub(r'[:\-\d]+.*$', '', cand_title).strip().lower()
                if len(t1) >= 4 and t1 == t2 and s_title.lower() != cand_title.lower():
                    overlap += 10
                    tag = "Direct Franchise Sequel" if ("2" in cand_title or "3" in cand_title) else "Franchise Connection"

            if s_data.get('director') and cand_dir and s_data['director'].lower() == cand_dir:
                overlap += 5
                tag = f"Director: {cand_row['director']}"
            if s_data.get('studio') and cand_studio and s_data['studio'].lower() == cand_studio:
                overlap += 4
                tag = f"Studio: {cand_row['studio'].title()}"

            shared_k = [k for k in s_data.get('keywords', set()).intersection(cand_k) if k not in STOP_WORDS]
            if shared_k:
                overlap += len(shared_k) * 2
                if not tag:
                    sorted_k = sorted(shared_k, key=len, reverse=True)
                    tag = ", ".join(k.title() for k in sorted_k[:2])

            shared_g = [g for g in s_data.get('genres', set()).intersection(cand_g) if g not in STOP_WORDS]
            if shared_g:
                overlap += len(shared_g)
                if not tag:
                    sorted_g = sorted(shared_g, key=lambda x: (x not in UMBRELLA_GENRES, len(x)), reverse=True)
                    tag = ", ".join(g.title() for g in sorted_g[:2])

            if overlap > best_overlap:
                best_overlap = overlap
                best_seed = s_data['title']
                best_match_tag = tag

        if best_seed and best_overlap > 0:
            tag_str = f" ({best_match_tag})" if best_match_tag else ""
            affinity = " + Community Co-Watch" if cf_score >= 0.35 else ""
            return f"Because you loved '{best_seed}'{tag_str}{affinity}"

    # CASE 2: Title Similarity (Option 2)
    matches = []
    target_title = rec_input.get("title", "") if isinstance(rec_input, dict) else ""
    cand_title = cand_row.get("title", "")
    if target_title and cand_title:
        t1 = re.sub(r'[:\-\d]+.*$', '', target_title).strip().lower()
        t2 = re.sub(r'[:\-\d]+.*$', '', cand_title).strip().lower()
        if len(t1) >= 4 and t1 == t2 and target_title.lower() != cand_title.lower():
            matches.append("Direct Franchise Sequel")

    t_dir = rec_input.get("director", "") if isinstance(rec_input, dict) else ""
    c_dir = cand_row.get("director") or ""
    if t_dir and c_dir and (t_dir.lower() in c_dir.lower() or c_dir.lower() in t_dir.lower()):
        matches.append(f"Director: {c_dir}")

    t_studio = rec_input.get("studio", "") if isinstance(rec_input, dict) else ""
    c_studio = cand_row.get("studio") or ""
    if t_studio and c_studio and t_studio.lower() == c_studio.lower():
        matches.append(f"Studio: {c_studio.replace('_', ' ').title()}")

    t_kw = set()
    if isinstance(rec_input, dict):
        kw_val = rec_input.get("keywords")
        if isinstance(kw_val, list):
            t_kw = set(k.lower().strip() for k in kw_val if k.strip())
        elif isinstance(kw_val, str):
            t_kw = set(k.lower().strip() for k in kw_val.split(',') if k.strip())
    
    c_kw = set(k.lower().strip() for k in (cand_row.get("keywords") or "").split(',') if k.strip())
    shared_kw = [k.title() for k in list(t_kw.intersection(c_kw))[:3] if k not in STOP_WORDS]
    if shared_kw:
        matches.extend(shared_kw)

    t_gen = set()
    if isinstance(rec_input, dict):
        g_val = rec_input.get("genres")
        if isinstance(g_val, list):
            t_gen = set(g.lower().strip() for g in g_val if g.strip())
        elif isinstance(g_val, str):
            t_gen = set(g.lower().strip() for g in g_val.split(',') if g.strip())
    c_gen = set(g.lower().strip() for g in (cand_row.get("genres") or "").split(',') if g.strip())
    shared_gen = [g.title() for g in list(t_gen.intersection(c_gen))[:2]]
    if shared_gen and len(matches) < 3:
        matches.extend(shared_gen)

    if cf_score >= 0.35:
        matches.append("Strong MovieLens Viewer Affinity")

    return ", ".join(matches[:4]) if matches else "Thematic Tone & Narrative Synergy"

def movie_recommender(db_path: str, user_id: str, recommendation_input, filter_watched: bool, filter_top_rank: bool) -> list[dict]:
    """
    Core Clustering-Driven Movie Recommendation Engine:
    Directly implements:
    1. Movie/Product recommendation using clustering
    2. Cluster users / items (Item K-Means & User Taste Cohorts)
    3. Dimensionality reduction (PCA) for latent space proximity
    """
    from modules.clustering import get_clustering_model
    cluster_model = get_clustering_model(db_path)

    # CASE A: Cold Start (User has zero watch history)
    if isinstance(recommendation_input, dict) and not recommendation_input.get("watched_movie_ids") and not recommendation_input.get("title") and not recommendation_input.get("movie_id"):
        # Select top exemplar film from each of the 8 clusters for maximum diversity
        results = []
        for c in range(cluster_model.n_clusters):
            c_df = cluster_model.df[cluster_model.df['cluster'] == c].sort_values(by=["pop", "final_score"], ascending=[False, False])
            if not c_df.empty:
                row = c_df.iloc[0]
                results.append({
                    "title": row["title"],
                    "year": row.get("release_year") or "",
                    "genres": row.get("genres") or "",
                    "rationale": f"Cluster #{c} Exemplar: {cluster_model.cluster_names[c]}",
                    "score": float(row["final_score"])
                })
        return results[:10]

    # CASE B: User Activity Taste Cohort Clustering (Option 1)
    if isinstance(recommendation_input, dict) and "watched_movie_ids" in recommendation_input:
        user_watched = recommendation_input.get("watched_movie_ids", [])
        user_pos_ids = recommendation_input.get("positive_movie_ids", [])

        if not user_pos_ids:
            # Fall back to diverse cluster exemplars excluding watched
            results = []
            for c in range(cluster_model.n_clusters):
                c_df = cluster_model.df[(cluster_model.df['cluster'] == c) & (~cluster_model.df['movie_id'].isin(user_watched))].sort_values(by=["pop", "final_score"], ascending=[False, False])
                if not c_df.empty:
                    row = c_df.iloc[0]
                    results.append({
                        "title": row["title"],
                        "year": row.get("release_year") or "",
                        "genres": row.get("genres") or "",
                        "rationale": f"Cluster #{c} Starter: {cluster_model.cluster_names[c]} (Rate movies >= 7.5 to personalize)",
                        "score": float(row["final_score"])
                    })
            return results[:10]

        recs = cluster_model.recommend_by_user_cluster(recommendation_input, top_n=10, filter_watched=filter_watched)
        return recs

    # CASE C: Item Clustering Recommendation (Option 2)
    target_movie_id = None
    if isinstance(recommendation_input, dict):
        target_movie_id = recommendation_input.get("movie_id")

    if target_movie_id:
        watched_ids = catch_watched_movies(db_path, user_id) if filter_watched else []
        recs = cluster_model.recommend_by_movie_cluster(target_movie_id, top_n=10, filter_watched_ids=watched_ids)
        return recs

    return []
