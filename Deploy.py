import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load movie dataset
movies_list = pd.read_csv("movies.csv")
movies_list_title = movies_list["title"].values

# Create a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_list['overview'].fillna(''))

# Convert TF-IDF matrix to sparse format
tfidf_matrix_sparse = csr_matrix(tfidf_matrix)

# Reduce dimensionality using TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Adjust the number of components
reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix_sparse)

# Calculate cosine similarity on the reduced matrix
similarity = cosine_similarity(reduced_tfidf_matrix, reduced_tfidf_matrix, dense_output=False)

def recommend(movie):
    movie_index = movies_list[movies_list["title"] == movie].index[0]
    distances = similarity[movie_index].flatten()
    sorted_movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []
    for i in sorted_movie_list:
        poster_path = movies_list["poster_path"].iloc[i[0]]
        recommended_movies.append(movies_list.iloc[i[0]].title)
        recommended_posters.append("https://image.tmdb.org/t/p/original" + poster_path)

    return recommended_movies, recommended_posters

st.title("Movie Recommendation System")

selected_movie_name = st.selectbox(
    "What's the movie name?",
    movies_list_title
)

if st.button("Recommend"):
    recommendation, movie_posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.write(recommendation[0])
        st.image(movie_posters[0])
    with col2:
        st.write(recommendation[1])
        st.image(movie_posters[1])
    with col3:
        st.write(recommendation[2])
        st.image(movie_posters[2])
    with col4:
        st.write(recommendation[3])
        st.image(movie_posters[3])
    with col5:
        st.write(recommendation[4])
        st.image(movie_posters[4])
