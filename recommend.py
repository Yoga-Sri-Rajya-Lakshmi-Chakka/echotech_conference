import streamlit as st
import pandas as pd
import requests
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude_vec2 = sum(a ** 2 for a in vec2) ** 0.5
    if magnitude_vec1 != 0 and magnitude_vec2 != 0:
        return dot_product / (magnitude_vec1 * magnitude_vec2)
    else:
        return 0  # Handle division by zero

def calculate_similarity(movie1, movie2):
    return cosine_similarity(movie1, movie2)

# Example recommendation function using Word2Vec similarities
def recommend(movie, movies_df, similarity_func, top_n=5):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    similarities = []
    for index, row in movies_df.iterrows():
        if index != movie_index:
            similarity = similarity_func(movies_df.loc[movie_index, 'tag_vectors'], row['tag_vectors'])
            similarities.append((row['title'], similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def convert_to_list(s):
    # Remove brackets and split by comma
    s = s.strip('[]')
    elements = s.split()
    # Convert elements to float
    return [float(element) for element in elements]


movies=pd.read_csv('new_movies1.csv',converters={'tag_vectors': convert_to_list})
st.markdown('<h1 style="text-align: center; color: red; font-family: Algerian;">Movie Recommendation System</h1>', unsafe_allow_html=True)
user_movie = st.text_input("Enter a movie title: ").title()
num_recommendations = st.number_input("How many recommendations do you want? ",min_value=1,max_value=10,step=1)
    
if st.button('Results'):
    recommendations = recommend(user_movie, movies, calculate_similarity, num_recommendations)
    for movie, similarity_score in recommendations:
        st.write(f"{movie} (Similarity Score: {similarity_score:.4f})")
    st.balloons()


