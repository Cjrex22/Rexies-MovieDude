# MovieDude ![Movie Recommendation System](https://img.shields.io/badge/Movie%20Recommendation%20System-orange?style=flat-square)

An advanced "content-based filtering" movie recommendation system built with Python, scikit-learn, and SQLite.<br>
It provides personalized movie suggestions based on user preferences through data analysis, and also allows users to search by a specific movie title or keyword to find similar recommendations.
<br>
The current demo utilizes a pre-processed dataset of the top 6,000 [TMDB](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) movies.

<img src="https://github.com/user-attachments/assets/cacf4628-27c0-4831-82a8-372857e6d378" width="800">

[Setup Requirements](#setup-requirements)  |  [Workflow Sections](#workflow-sections)  |  [DataBase Structure](#database-structure) | [MovieDude on DeepWiki](https://deepwiki.com/a-partovii/MovieDude)

---
## 🚀 Quick Start (Run in 3 Steps)

### Step 1: Open terminal in the project folder
```bash
cd movieml
```

### Step 2: Install required libraries (run once)
```bash
pip install -r requirements.txt
```
*(Recommended: you can also create a virtual environment first: `python3 -m venv .venv && source .venv/bin/activate` on Mac/Linux or `.venv\Scripts\activate` on Windows)*

### Step 3: Start the application
```bash
python3 src/main.py
```
*(On Windows, use `python src/main.py`)*

### 🔑 Login Credentials:
- **User ID:** `Aryan`
- **Password:** `12345`

*(Aryan's account is pre-configured with active ratings for instant personalized recommendations)*

---
### Workflow Sections:
- <strong>On Users Table Flow </strong>>>><br>
  Extract and analyze user activity data from the database to determine personal preferences.
  <ol type="1">
    <li>Apply Min–Max data normalization</li>
    <li>Extract top favorite movies by normalized score</li>
    <li>Use quasi-NLP to extract movie attributes</li>
    <li>Return final result processed matrix ready for model use</li>
  </ol>
- <strong>Main Engine Processes</strong> >>><br>
  Performs similarity analysis and recommendation generation based on user preferences and the database contents.

  <ol type="1">
    <li>Apply selected options #1 (filter out watched movies)</li>
    <li>Pre-processing validation check</li>
    <li>Multi-binary encoding</li>
    <li>Execute engine process</li>
    <li>Apply selected options #2 (filter for high-rated movies)</li>
    <li>Return final result as a list (array)</li>
  </ol>
  
---
### DataBase Structure:
```
MovieDude.db
│
├── TABLE: Movies
│   ├── movie_id: INTEGER (PRIMARY KEY)
│   ├── title: TEXT (NOT NULL)
│   ├── release_year: INTEGER
│   ├── genres: TEXT (Split by comma ",")
│   ├── original_lang: VARCHAR(20)
│   ├── director: TEXT (Split by comma ",")
│   ├── stars: TEXT (Split by comma ",")
│   ├── keywords: TEXT (Split by comma ",")
│   ├── rating: REAL
│   ├── rating_count: INTEGER
│   └── final_score: REAL
│ 
├── TABLE: Users
│   ├── user_id: VARCHAR(20) (PRIMARY KEY)
│   ├── password: VARCHAR(20) (Prototype)
│   ├── name: VARCHAR(40)
│   └── created_at: TIMESTAMP
│
└── TABLE: Users_data
    ├── user_id: VARCHAR(20)
    ├── movie_id: INT
    ├── user_rate: REAL
    └── liked: BOOLEAN (DEFAULT FALSE)

```

---

**For more details, follow: [deepwiki.com/a-partovii/MovieDude](https://deepwiki.com/a-partovii/MovieDude)** &nbsp; [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/a-partovii/MovieDude)
