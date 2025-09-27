import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Load The Model & Data ---
try:
    movie_fingerprints = joblib.load('movie_fingerprints.joblib')
    df = pd.read_pickle('processed_data.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run the Colab notebook first to generate them.")
    st.stop()

# --- 2. Gothic UI Configuration ---
st.set_page_config(page_title="Cine-Seance", layout="wide", page_icon="🔮")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;700&display=swap');
html, body, [class*="st-"] { font-family: 'Crimson Text', serif; }
.stApp { background-color: #1a1a1a; color: #e0e0e0; }
h1, h2, h3 { color: #9a4ca6; }
.stButton>button { border: 2px solid #5c2c69; background-color: transparent; color: #e0e0e0; padding: 0.25em 0.5em; font-size: 14px; margin: 2px; transition: all 0.3s ease-in-out; }
.stButton>button:hover { border-color: #c789d6; background-color: #5c2c69; color: #ffffff; }
div[data-testid*="stHorizontalBlock"] > div[data-testid*="stButton"] > button { border: none !important; padding: 0 !important; color: #5c2c69; background-color: transparent !important; }
div[data-testid*="stHorizontalBlock"] > div[data-testid*="stButton"] > button:hover { color: #fca311; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Cine-Seance")
st.markdown("Summon spirits of the silver screen. Rate films you have witnessed to reveal your unique taste profile.")

# --- 3. Session State Initialization ---
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = {} # Store as {movie_index: rating}
    st.session_state.movie_pool = df.sample(10)
    st.session_state.rated_indices = set()

# --- 4. Interactive Rating Grid ---
st.markdown("---")
st.subheader("First, channel your tastes by rating these films:")

grid_cols = st.columns(2)
col_index = 0
for index, movie in st.session_state.movie_pool.iterrows():
    with grid_cols[col_index % 2]:
        st.markdown(f"**{movie['title']}**")
        st.caption(f"_{', '.join(movie['genres_list'])}_")
        
        star_cols = st.columns(5)
        for i in range(5):
            if star_cols[i].button("⭐", key=f"star_{movie['id']}_{i+1}"):
                st.session_state.rated_movies[index] = i + 1
                st.session_state.rated_indices.add(index)
                st.toast(f"You rated '{movie['title']}' {i+1} stars.")
        st.markdown("<br>", unsafe_allow_html=True)
    col_index += 1

st.markdown("---")

# --- 5. Control Buttons & Recommendation Logic ---
control_cols = st.columns([1, 2])
if control_cols[0].button("Reveal More Films 🔄"):
    available_movies = df[~df.index.isin(st.session_state.rated_indices)]
    st.session_state.movie_pool = available_movies.sample(min(10, len(available_movies)))
    st.rerun()

if control_cols[1].button("Conjure My Recommendations ✨", type="primary"):
    if len(st.session_state.rated_movies) < 3:
        st.warning("You must rate at least 3 films to perform the seance.")
    else:
        with st.spinner("Analyzing your cinematic spirit..."):
            rated_indices = list(st.session_state.rated_movies.keys())
            ratings = np.array(list(st.session_state.rated_movies.values()))
            
            # Get the fingerprints of the movies the user rated
            rated_fingerprints = movie_fingerprints[rated_indices]
            
            # Create the user's taste profile via weighted average
            user_taste_profile = np.dot(ratings, rated_fingerprints) / ratings.sum()
            
            # Calculate cosine similarity between the user's profile and all movies
            similarities = cosine_similarity(user_taste_profile.reshape(1, -1), movie_fingerprints)[0]
            
            # Get the indices of the most similar movies
            similar_movie_indices = np.argsort(similarities)[::-1]
            
            st.subheader("The spirits have spoken! You may enjoy these apparitions:")
            rec_count = 0
            for movie_index in similar_movie_indices:
                if rec_count >= 10: break
                if movie_index not in st.session_state.rated_indices:
                    rec_movie_info = df.iloc[movie_index]
                    st.markdown(f"**{rec_count+1}. {rec_movie_info['title']}** — _{', '.join(rec_movie_info['genres_list'])}_")
                    rec_count += 1

# --- 6. Sidebar for User's Ratings ---
with st.sidebar:
    st.header("Your Ratings")
    if not st.session_state.rated_movies:
        st.info("Your rating history is empty.")
    else:
        for index, rating in st.session_state.rated_movies.items():
            title = df.loc[index, 'title']
            st.markdown(f"- **{title}:** {'⭐' * rating}")
    if st.session_state.rated_movies and st.button("Clear All Ratings"):
        st.session_state.rated_movies = {}
        st.session_state.rated_indices = set()
        st.rerun()
