from __future__ import annotations
import re
import sqlite3
import difflib
import unicodedata
try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

def _print_msg(msg, style=None):
    if console and style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)

def _strip_diacritics(text: str) -> str:
    """Converts accented/macron characters to plain ASCII (e.g. 'Bāhubali' -> 'Bahubali', 'Amélie' -> 'Amelie')."""
    if not text:
        return ""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def _clean_article(text: str) -> str:
    """Strips leading English articles for robust title matching."""
    return re.sub(r'^(the|a|an)\s+', '', text.lower().strip())

def _extract_features(genres_raw, keywords_raw):
    genres = [g.strip().lower() for g in (genres_raw or "").split(',') if g.strip()]
    keywords = [k.strip().lower() for k in (keywords_raw or "").split(',') if k.strip()]
    return [genres, keywords]

LANG_MAP = {
    'hi': 'Hindi', 'ml': 'Malayalam', 'ta': 'Tamil', 'te': 'Telugu',
    'en': 'English', 'ja': 'Japanese', 'ko': 'Korean', 'fr': 'French',
    'es': 'Spanish', 'de': 'German', 'it': 'Italian', 'zh': 'Chinese',
    'cn': 'Chinese', 'ru': 'Russian', 'pt': 'Portuguese'
}

ALIASES = {
    "ddlj": "Dilwale Dulhania Le Jayenge",
    "znmd": "Zindagi Na Milegi Dobara",
    "yjhd": "Yeh Jawaani Hai Deewani",
    "k3g": "Kabhi Khushi Kabhie Gham",
    "khnh": "Kal Ho Naa Ho",
    "rdb": "Rang De Basanti",
    "gow": "Gangs of Wasseypur",
    "kgf": "K.G.F: Chapter 1",
    "kgf 1": "K.G.F: Chapter 1",
    "kgf 2": "K.G.F: Chapter 2",
    "k.g.f": "K.G.F: Chapter 1",
    "dune 2": "Dune: Part Two",
    "deadpool 3": "Deadpool & Wolverine",
    "spiderverse": "Spider-Man: Across the Spider-Verse",
    "across the spiderverse": "Spider-Man: Across the Spider-Verse",
    "three idiots": "3 Idiots",
    "brahmastra": "Brahmāstra: Part One – Shiva",
    "pushpa": "Pushpa: The Rise",
    "pushpa 1": "Pushpa: The Rise",
    "pushpa 2": "Pushpa 2: The Rule",
    "salaar": "Salaar: Part 1 – Ceasefire",
    "kalki": "Kalki 2898 AD",
    "ms dhoni": "M.S. Dhoni: The Untold Story",
    "dhoni": "M.S. Dhoni: The Untold Story",
    "m.s. dhoni": "M.S. Dhoni: The Untold Story",
    "bhag milkha bhag": "Bhaag Milkha Bhaag",
    "bhaag milkha bhaag": "Bhaag Milkha Bhaag",
    "milkha": "Bhaag Milkha Bhaag",
    "pursuit of happiness": "The Pursuit of Happyness",
    "the pursuit of happiness": "The Pursuit of Happyness",
    "bahubali 1": "Bāhubali: The Beginning",
    "baahubali 1": "Bāhubali: The Beginning",
    "bhaubali 1": "Bāhubali: The Beginning",
    "bahubali 2": "Bāhubali 2: The Conclusion",
    "baahubali 2": "Bāhubali 2: The Conclusion",
    "bhaubali 2": "Bāhubali 2: The Conclusion",
    "bahubali the beginning": "Bāhubali: The Beginning",
    "baahubali the beginning": "Bāhubali: The Beginning",
    "bhaubali the beginning": "Bāhubali: The Beginning",
    "bahubali the conclusion": "Bāhubali 2: The Conclusion",
    "baahubali the conclusion": "Bāhubali 2: The Conclusion",
    "bhaubali the conclusion": "Bāhubali 2: The Conclusion",
    "bahubali 2 the conclusion": "Bāhubali 2: The Conclusion",
    "baahubali 2 the conclusion": "Bāhubali 2: The Conclusion",
    "bhaubali 2 the conclusion": "Bāhubali 2: The Conclusion",
    "baahubali": "bahubali",
    "bhaubali": "bahubali",
}

def _resolve_choice(choice, close_matches):
    """
    Friction-free choice resolver:
    Accepts:
    - Index number ('1', '2', etc.)
    - Exact or lowercase title ('the pursuit of happyness')
    - Substring / key word ('happyness', 'malayalam')
    - Affirmative shortcut ('y', 'yes') when 1 match
    - Empty Enter when 1 match
    - Fuzzy match against candidate list
    """
    choice = choice.strip()
    if not choice:
        if len(close_matches) == 1:
            return close_matches[0]
        return None

    # 1. Back / Cancel shortcut
    if choice.lower() in ("b", "back", "cancel", "n", "no"):
        return None

    # 2. Direct index number
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(close_matches):
            return close_matches[idx - 1]
        return None

    # 3. Affirmative shortcut
    if choice.lower() in ("y", "yes") and len(close_matches) == 1:
        return close_matches[0]

    choice_lower = choice.lower()

    # 4. Exact case-insensitive match
    for m in close_matches:
        if choice_lower == m.lower():
            return m

    # 5. Substring match
    for m in close_matches:
        if choice_lower in m.lower() or m.lower() in choice_lower:
            return m

    # 6. Fuzzy match among suggestions
    fuzzy = difflib.get_close_matches(choice, close_matches, n=1, cutoff=0.5)
    if fuzzy:
        return fuzzy[0]

    return None

def _format_movie_entry(title: str, m_dict: dict) -> str:
    year = f" ({m_dict['year']})" if m_dict.get('year') else ""
    lang_code = m_dict.get('lang') or ''
    lang_name = LANG_MAP.get(lang_code, lang_code.upper() if lang_code else '')
    lang_str = f", {lang_name}" if lang_name else ""
    genres = f", {m_dict['genres'].split(',')[0].strip()}" if m_dict.get('genres') else ""
    return f"{title}{year}{lang_str}{genres}"

def find_by_title(db_path, title):
    """
    Retrieves full movie metadata by title.
    Features:
    - Alias lookup (DDLJ, ZNMD, pursuit of happiness, etc.)
    - Exact match & disambiguation handling (e.g. Hindi vs Malayalam versions)
    - Normalized article matching ('dark knight' -> 'The Dark Knight')
    - Fuzzy 'Did you mean' suggestions with zero friction (accepts index or typed title)
    - Graceful out-of-vocabulary fallback with helpful suggestions
    """
    raw_query = title.strip()
    clean_title = raw_query.lower()
    if clean_title in ALIASES:
        clean_title = ALIASES[clean_title].lower()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Load catalogue metadata dictionary
            cursor.execute("""
            SELECT movie_id, title, genres, keywords, director, stars, studio, country, release_year, orginall_lang
            FROM Movies_sorted
            """)
            all_rows = cursor.fetchall()

            all_movies = {}
            for r in all_rows:
                all_movies[r[1]] = {
                    "movie_id": r[0],
                    "title": r[1],
                    "genres": r[2] or "",
                    "keywords": r[3] or "",
                    "director": r[4] or "",
                    "stars": r[5] or "",
                    "studio": r[6] or "",
                    "country": r[7] or "",
                    "year": r[8] or "",
                    "lang": r[9] or ""
                }

            # Normalize query with diacritics stripped
            q_norm = _strip_diacritics(clean_title).lower().strip()
            q_no_art = _clean_article(q_norm)

            # 1. Exact match on full title or core title before subtitle
            exact_matches = [t for t in all_movies if _strip_diacritics(t).lower().strip() == q_norm]
            core_matches = [t for t in all_movies if _strip_diacritics(t.split(':')[0]).lower().strip() == q_norm or _strip_diacritics(t.split('–')[0]).lower().strip() == q_norm]

            # Check for version disambiguation companions (e.g. Drishyam -> Drishyam (Malayalam), Bahubali -> Bahubali 1 & 2)
            companions = [
                t for t in all_movies
                if _strip_diacritics(t).lower().startswith(q_norm + " (")
                or _strip_diacritics(t.split(':')[0]).lower().strip() == q_norm + " 2"
                or _strip_diacritics(t.split(':')[0]).lower().strip() == q_norm + " 1"
                or (_strip_diacritics(t).lower().startswith(q_norm + ":") and t not in exact_matches and t not in core_matches)
            ]

            all_exact_candidates = list(dict.fromkeys(exact_matches + core_matches))

            # If multiple versions exist, present disambiguation prompt
            if ((exact_matches or core_matches) and companions) or len(all_exact_candidates) > 1:
                disambig_options = list(dict.fromkeys(all_exact_candidates + companions))
                if len(disambig_options) > 1:
                    _print_msg(f"\nFound multiple versions for '{raw_query}':", style="cyan")
                    for idx, opt in enumerate(disambig_options, 1):
                        entry_str = _format_movie_entry(opt, all_movies[opt])
                        print(f"  {idx}. {entry_str}")

                    choice = input("\nEnter number to choose (or type title, press Enter for #1): ").strip()
                    resolved = _resolve_choice(choice, disambig_options) or disambig_options[0]
                    m_data = all_movies[resolved]
                    _print_msg(f"✅ Selected: {_format_movie_entry(resolved, m_data)}", style="green")
                    genres, keywords = _extract_features(m_data["genres"], m_data["keywords"])
                    m_data["genres"] = genres
                    m_data["keywords"] = keywords
                    return m_data

            if len(all_exact_candidates) == 1:
                selected_title = all_exact_candidates[0]
                m_data = all_movies[selected_title]
                _print_msg(f"✅ Found: {_format_movie_entry(selected_title, m_data)}", style="green")
                genres, keywords = _extract_features(m_data["genres"], m_data["keywords"])
                m_data["genres"] = genres
                m_data["keywords"] = keywords
                return m_data

            # 2. Check for article-stripped exact match (e.g. 'dark knight' -> 'The Dark Knight')
            art_matches = [t for t in all_movies if _clean_article(_strip_diacritics(t)) == q_no_art]
            if len(art_matches) == 1:
                selected_title = art_matches[0]
                m_data = all_movies[selected_title]
                _print_msg(f"✅ Found: {_format_movie_entry(selected_title, m_data)}", style="green")
                genres, keywords = _extract_features(m_data["genres"], m_data["keywords"])
                m_data["genres"] = genres
                m_data["keywords"] = keywords
                return m_data

            # 3. Check for substring / LIKE matches
            like_matches = [t for t in all_movies if q_norm in _strip_diacritics(t).lower()]
            if len(like_matches) == 1:
                selected_title = like_matches[0]
                m_data = all_movies[selected_title]
                _print_msg(f"✅ Found: {_format_movie_entry(selected_title, m_data)}", style="green")
                genres, keywords = _extract_features(m_data["genres"], m_data["keywords"])
                m_data["genres"] = genres
                m_data["keywords"] = keywords
                return m_data

            # 4. Fuzzy matches on core title and full title with adaptive cutoff
            fuzzy_candidates = []
            for t in all_movies:
                t_norm = _strip_diacritics(t).lower()
                core_t = _strip_diacritics(t.split(':')[0]).lower().strip()
                r_core = difflib.SequenceMatcher(None, q_norm, core_t).ratio()
                r_full = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()
                best_r = max(r_core, r_full)
                if best_r >= 0.70:
                    fuzzy_candidates.append((t, best_r))

            fuzzy_candidates.sort(key=lambda x: x[1], reverse=True)
            close_matches = [t for t, _ in fuzzy_candidates[:5]]
            if not close_matches and like_matches and len(like_matches) <= 5:
                close_matches = like_matches[:5]

            if close_matches:
                _print_msg(f"\n⚠️  No exact match found for '{raw_query}'. Did you mean:", style="yellow")
                for idx, match in enumerate(close_matches, 1):
                    entry_str = _format_movie_entry(match, all_movies[match])
                    print(f"  {idx}. {entry_str}")

                choice = input("\nEnter number to choose (or type title, press Enter for #1): ").strip()
                resolved = _resolve_choice(choice, close_matches)
                if resolved:
                    m_data = all_movies[resolved]
                    _print_msg(f"✅ Selected: {_format_movie_entry(resolved, m_data)}", style="green")
                    genres, keywords = _extract_features(m_data["genres"], m_data["keywords"])
                    m_data["genres"] = genres
                    m_data["keywords"] = keywords
                    return m_data
                else:
                    _print_msg("↩️ Returning to search prompt.\n", style="dim")
                    return None
            else:
                _print_msg(f"❌ No matching movie found for '{raw_query}' in the current catalogue (~6,100 released titles).", style="red")
                if "hail mary" in clean_title or "project hail mary" in clean_title:
                    print("💡 Tip: 'Project Hail Mary' is an upcoming / unreleased film. Try released sci-fi classics like 'Interstellar', 'The Martian', 'Arrival', or 'Contact'.\n")
                else:
                    print("💡 Tip: Check your spelling or try popular titles like 'Inception', 'Toy Story', 'Drishyam', or 'Fight Club'.\n")
                return None

    except sqlite3.Error as error:
        print(f"❌ SQLite error: {error}")
        return None
