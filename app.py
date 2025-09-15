# app.py
import streamlit as st
import pandas as pd
import joblib
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
import os

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# -------------------------
# Load models & data
# -------------------------
@st.cache_data
def load_models_and_data():
    models = {}
    models['User-KNN'] = joblib.load('saved_models/algo_user_knn.joblib')
    models['Item-KNN'] = joblib.load('saved_models/algo_item_knn.joblib')
    models['SVD']      = joblib.load('saved_models/algo_svd.joblib')
    movies_df = pd.read_csv('saved_models/movies_clean.csv', dtype={'movieId': str})
    ratings_df = pd.read_csv('ratings.csv', dtype={'userId': str, 'movieId': str})
    if "timestamp" in ratings_df.columns:
        ratings_df = ratings_df.drop(columns=["timestamp"])
    return models, movies_df, ratings_df

models, movies_df, ratings = load_models_and_data()

st.title("🎬 Movie Recommendation System (MovieLens)")

# -------------------------
# Sidebar options
# -------------------------
st.sidebar.header("Options")
user_id = st.sidebar.text_input("Enter user id", value=str(ratings['userId'].iloc[0]))
model_choice = st.sidebar.selectbox("Model", options=['User-KNN', 'Item-KNN', 'SVD'])
top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=30, value=10)

st.write(f"Selected model: **{model_choice}** — Top **{top_n}** recommendations for user **{user_id}**")

# -------------------------
# Build trainset
# -------------------------
@st.cache_data
def build_trainset_for_loaded_models(ratings):
    reader = Reader(rating_scale=(ratings['rating'].min(), ratings['rating'].max()))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    return trainset

trainset = build_trainset_for_loaded_models(ratings)

# -------------------------
# Recommendation function
# -------------------------
def get_top_n(algo, trainset, movies_df, user_raw_id, n=10):
    try:
        # Map raw user id to inner id
        inner_uid = trainset.to_inner_uid(user_raw_id)
    except ValueError:
        # Cold-start fallback → recommend top popular movies
        top_pop = ratings.groupby("movieId")["rating"].mean().sort_values(ascending=False).head(n).reset_index()
        top_pop = top_pop.merge(movies_df, on="movieId", how="left")
        top_pop = top_pop.rename(columns={"rating": "pred_rating"})
        return top_pop

    # Movies the user has already rated
    user_rated_inner = set([iid for (iid, _) in trainset.ur[inner_uid]])
    user_rated_raw = set([trainset.to_raw_iid(iid) for iid in user_rated_inner])

    # Candidate movies (not yet rated by the user)
    all_movie_ids = movies_df['movieId'].unique().tolist()
    candidates = [mid for mid in all_movie_ids if mid not in user_rated_raw]

    # Predict ratings for candidates
    preds = []
    for mid in candidates:
        p = algo.predict(user_raw_id, mid)
        preds.append((mid, p.est))

    # Sort predictions and get top N
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
    df_top = pd.DataFrame(preds_sorted, columns=['movieId', 'pred_rating'])
    df_top = df_top.merge(movies_df[['movieId', 'title', 'genres']], on='movieId', how='left')
    return df_top

# -------------------------
# Show recommendations
# -------------------------
if st.button("Get recommendations"):
    algo = models[model_choice]
    topdf = get_top_n(algo, trainset, movies_df, user_id, n=top_n)
    topdf = topdf[['title', 'pred_rating', 'genres']]  # keep only the needed columns

    # Reset index to start from 1 instead of 0
    topdf.index = range(1, len(topdf) + 1)

    st.write("### Recommended Movies")
    st.dataframe(topdf)