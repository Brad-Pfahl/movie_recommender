from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and process data once on startup
movies_path = os.path.join("data", "movies_metadata.csv")
df = pd.read_csv(movies_path, low_memory=False)
df['overview'] = df['overview'].fillna('')

def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [genre['name'] for genre in genres]
    except (ValueError, SyntaxError):
        return []

df['genre_list'] = df['genres'].apply(parse_genres)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

@app.get("/genres")
def get_genres():
    from itertools import chain
    all_genres = sorted(set(chain.from_iterable(df['genre_list'])))
    return {"genres": all_genres}

@app.get("/movies")
def get_movies_by_genre(genre: str, limit: int = 10):
    matches = df[df['genre_list'].apply(lambda x: genre in x)]
    top = matches[['title', 'overview']].head(limit).to_dict(orient='records')
    return {"movies": top}

@app.get("/recommend")
def get_recommendations(title: str, top_n: int = 10):
    idx = indices.get(title)
    if idx is None:
        return {"error": f"Movie titled '{title}' not found."}
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    recs = df[['title', 'overview']].iloc[movie_indices].to_dict(orient='records')
    return {"recommendations": recs}
