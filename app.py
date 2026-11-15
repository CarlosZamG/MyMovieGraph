import torch
torch.classes.__path__ = []  # Prevents Streamlit-PyTorch inspection crashes

import streamlit as st
import pandas as pd
import traceback
import json

from recommend_new_user_fast import load_fast_resources, recommend_for_new_user_fast

st.set_page_config(
    page_title="GraphSAGE Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 My MovieGraph 🎞️")
st.markdown("Search for movies you have watched, rate them using stars ⭐, and get personalized recommendations.")

# --- 1. Load Cached Resources (Runs Once and Persists in RAM) ---
@st.cache_resource
def get_fast_resources():
    return load_fast_resources(
        base_bundle_dir="inference_bundle",
        fast_bundle_dir="fast_inference_bundle",
        device="cpu",
    )

with st.spinner("Loading recommendation model..."):
    try:
        resources = get_fast_resources()
        movies_df = resources["movies_df"]
        #movies_df, ratings_df, movie_id_to_mapped, mapped_to_movie_id, movie_features_cpu, checkpoint = get_static_resources()
        #preloaded_resources = (movies_df, ratings_df, movie_id_to_mapped, mapped_to_movie_id, movie_features_cpu, checkpoint)
        
        # Create a fast O(1) lookup dictionary for movie titles to speed up rendering
        if "movie_title_map" not in st.session_state:
            st.session_state.movie_title_map = movies_df.set_index('movieId')['title'].to_dict()
            
    except Exception as e:
        st.error("Failed to load static graph resources:")
        st.code(traceback.format_exc(), language="python")
        st.stop()

# --- 2. Initialize Session State for User Ratings ---
if "user_ratings" not in st.session_state:
    st.session_state.user_ratings = {}  # Format: {movieId: rating}

# --- 3. Search & Rate Movies ---
st.header("➕ Add Movie Ratings")
st.markdown("Search the MovieLens database and assign ratings:")

# Create side-by-side columns for search and selection
col_search, col_select = st.columns([1, 2], vertical_alignment="center")

with col_search:
    search_query = st.text_input("Search Movie Title", placeholder="e.g., Toy Story, Batman...")

selected_movie_id = None
movie_options = {}

if search_query and len(search_query.strip()) >= 2:
    matches = movies_df[movies_df['title'].str.contains(search_query, case=False, regex=False, na=False)]
    if not matches.empty:
        movie_options = {f"{row['title']} (ID: {row['movieId']})": int(row['movieId']) for _, row in matches.head(20).iterrows()}

with col_select:
    if movie_options:
        selected_label = st.selectbox("Select from matches:", list(movie_options.keys()))
        if selected_label:
            selected_movie_id = movie_options[selected_label]
    else:
        st.selectbox("Select from matches:", ["Type at least 2 characters to search..."], disabled=True)

if search_query and len(search_query.strip()) >= 2 and not movie_options:
    st.info("No matching movies found.")

movie_title = ""
if selected_movie_id is not None:
    movie_title = st.session_state.movie_title_map.get(selected_movie_id, "")

st.subheader(f"Rate: {movie_title if movie_title else 'Select a movie'}")

rating_options = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

rating = st.select_slider(
    "Rate this movie",
    options=rating_options,
    value=3.0,
    format_func=lambda x: f"{x} ⭐",
    key=f"select_rating_{selected_movie_id}"
)

if selected_movie_id:
    col1, col2 = st.columns([1, 3], vertical_alignment="center")

    with col1:
        submitted = st.button("Save Rating", key=f"save_{selected_movie_id}")

    if submitted:
        st.session_state.user_ratings[selected_movie_id] = rating
        with col2:
            st.success(f"Saved {rating}-star rating for **{movie_title}**!")


# --- 3.5 Import / Export Ratings File (Optimized Guard) ---
with st.expander("📁 Backup & Restore Ratings (JSON File)"):
    st.markdown("Download your current ratings dictionary as a backup file, or upload an existing ratings file to load them instantly.")
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        ratings_json = json.dumps(st.session_state.user_ratings, indent=4)
        st.download_button(
            label="📥 Export Ratings to File",
            data=ratings_json,
            file_name="my_movie_ratings.json",
            mime="application/json",
            disabled=len(st.session_state.user_ratings) == 0,
            help="Download your rating dictionary as a plain text JSON file"
        )
        
    with exp_col2:
        uploaded_file = st.file_uploader("📤 Upload Ratings File", type=["json", "txt"], key="rating_uploader")
        if uploaded_file is not None:
            # Generate a unique fingerprint for the file to prevent infinite re-parsing loops
            file_fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get("last_processed_file") != file_fingerprint:
                try:
                    content = uploaded_file.read().decode("utf-8")
                    loaded_dict = json.loads(content)
                    
                    st.session_state.user_ratings = {int(k): float(v) for k, v in loaded_dict.items()}
                    st.session_state.recommendations = None
                    st.session_state["last_processed_file"] = file_fingerprint
                    st.success(f"Successfully loaded {len(st.session_state.user_ratings)} ratings!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to parse ratings file: {ex}")


# --- 4. Main Panel: Display Current Profile ---
st.subheader("📋 Your Watch & Rating Profile 🍿")

if st.session_state.user_ratings:
    profile_data = []
    title_map = st.session_state.movie_title_map
    
    # Fast O(1) dictionary lookups instead of slow DataFrame filtering
    for m_id, rat in st.session_state.user_ratings.items():
        title = title_map.get(m_id, f"Movie ID {m_id}")
        profile_data.append({"Title": title, "Rating": f"{rat} ⭐"})

    st.dataframe(pd.DataFrame(profile_data), width="stretch")

    if st.button("🗑️ Clear All Ratings"):
        st.session_state.user_ratings = {}
        st.session_state.recommendations = None
        st.session_state.pop("last_processed_file", None) # Clear file tracker
        st.rerun()
else:
    st.info("Your profile is empty. Use the searchbox or upload a backup file above.")

# --- 5. Generate Recommendations Section ---
st.markdown("---")
st.subheader("🚀 Generate Predictions")

top_k = st.slider("Number of Recommendations (Top-K)", min_value=5, max_value=30, value=10, key="top_k_slider")

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

run_clicked = st.button("✨ Run GraphSAGE Recommendations", type="primary", disabled=len(st.session_state.user_ratings) == 0)

if run_clicked:
    with st.spinner("Executing GraphSAGE message passing and LinkNeighborLoader evaluation..."):
        try:
            st.session_state.recommendations = recommendations_df = recommend_for_new_user_fast(
                                                new_user_ratings=st.session_state.user_ratings,
                                                top_k=top_k,
                                                device="cpu",
                                                preloaded_resources=resources
                                            )
        except Exception as e:
            st.error("An error occurred while generating recommendations:")
            st.code(traceback.format_exc(), language="python")
            st.session_state.recommendations = None

if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
    st.success("Recommendations active! Save ratings for any movies you like.")

    for _, row in st.session_state.recommendations.iterrows():
        rec_movie_id = int(row['movieId'])
        rec_title = row['title']

        with st.container():
            r_col1, r_col2, r_col3 = st.columns([3, 1, 1])
            with r_col1:
                st.markdown(f"### {rec_title}")
                st.caption(f"**Genres:** {row['genres']}")
            with r_col2:
                st.metric(label="Model Score", value=f"{row['score']:.4f}")
            with r_col3:
                st.markdown(f"**Movie ID:** `{rec_movie_id}`")

            rate_col1, rate_col2 = st.columns([3, 1], vertical_alignment="center")

            with rate_col1:
                rec_rating = st.select_slider(
                    "Rate this recommendation",
                    options=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                    value=3.0,
                    format_func=lambda x: f"{x} ⭐",
                    key=f"rec_slider_{rec_movie_id}"
                )

            with rate_col2:
                if st.button("Save Rating", key=f"rec_save_{rec_movie_id}"):
                    st.session_state.user_ratings[rec_movie_id] = rec_rating
                    st.toast(f"Saved {rec_rating}-star rating for '{rec_title}'!", icon="⭐")

            st.markdown("---")