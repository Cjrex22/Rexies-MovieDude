import sqlite3
import pandas as pd
from collections import Counter
from itertools import chain

def find_user_interests(db_path: str, user_id: str) -> dict:
    """
    Extracts rich user profile interests from Users_data:
    - Watched movie IDs (for negative filtering / exclusion)
    - Positive rated movie IDs: STRICTLY user_rate >= 7.5
      (Any movie rated < 7.5 is excluded from positive seeds and taste profile)
    - Dictionary of positive movie metadata for explainable rationales
    - Top genres, keywords, directors, studios, and countries
    """
    conn = sqlite3.connect(db_path)
    query = '''
    SELECT ud.user_id, ud.movie_id, ud.user_rate, ud.liked, ms.title, ms.genres, ms.keywords, ms.director, ms.stars, ms.studio, ms.country
    FROM Users_data ud
    JOIN Movies_sorted ms ON ud.movie_id = ms.movie_id
    WHERE ud.user_id = ?
    '''
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()

    if df.empty:
        return {}

    watched_movie_ids = list(df["movie_id"].unique())

    # STRICT THRESHOLD: Only consider movies rated >= 7.5
    pos_df = df[df["user_rate"] >= 7.5]
    if pos_df.empty:
        return {
            "user_id": user_id,
            "watched_movie_ids": watched_movie_ids,
            "positive_movie_ids": [],
            "pos_movies_dict": {},
            "genres": [],
            "keywords": [],
            "director": "",
            "studio": "",
            "country": "",
            "ratings": {}
        }

    positive_movie_ids = list(pos_df["movie_id"].unique())

    pos_movies_dict = {}
    for _, row in pos_df.iterrows():
        pos_movies_dict[row["movie_id"]] = {
            "title": row["title"],
            "director": row["director"] or "",
            "studio": row["studio"] or "",
            "genres": set(g.strip().lower() for g in (row["genres"] or "").split(',') if g.strip()),
            "keywords": set(k.strip().lower() for k in (row["keywords"] or "").split(',') if k.strip())
        }

    # Extract top genres & keywords
    genres_series = pos_df["genres"].dropna().str.lower().str.split(", ")
    keywords_series = pos_df["keywords"].dropna().str.lower().str.split(", ")
    top_genres = [g[0] for g in Counter(chain.from_iterable(genres_series)).most_common(6)]
    top_keywords = [k[0] for k in Counter(chain.from_iterable(keywords_series)).most_common(10)]

    directors = [d.strip() for d in pos_df["director"].dropna() if d.strip()]
    top_directors = [d[0] for d in Counter(directors).most_common(3)]

    studios = [s.strip() for s in pos_df["studio"].dropna() if s.strip()]
    top_studios = [s[0] for s in Counter(studios).most_common(2)]

    countries = [c.strip() for c in pos_df["country"].dropna() if c.strip()]
    top_countries = [c[0] for c in Counter(countries).most_common(2)]

    return {
        "user_id": user_id,
        "watched_movie_ids": watched_movie_ids,
        "positive_movie_ids": positive_movie_ids,
        "pos_movies_dict": pos_movies_dict,
        "genres": top_genres,
        "keywords": top_keywords,
        "director": ", ".join(top_directors),
        "studio": ", ".join(top_studios),
        "country": ", ".join(top_countries),
        "ratings": dict(zip(pos_df["movie_id"], pos_df["user_rate"]))
    }
