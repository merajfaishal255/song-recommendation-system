import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

st.title("🎵 Song Recommendation System")

# Load dataset
df = pd.read_csv("data.csv")

# Combine mood + energy into one text feature
df["features"] = df["mood"] + " " + df["energy"]
vectorizer = CountVectorizer()
vector_matrix = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(vector_matrix)

# User input
song_name = st.selectbox("Choose a song you like:", df["song"].values)

def recommend(song):
    index = df[df["song"] == song].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_songs = sorted(distances, key=lambda x: x[1], reverse=True)[1:4]

    recommended_list = []
    for i in sorted_songs:
        recommended_list.append(df.iloc[i[0]]["song"])
    return recommended_list

if st.button("Recommend"):
    st.subheader("🎧 Recommended Songs:")
    recs = recommend(song_name)

    for s in recs:
        st.write(f"▶️ **{s}**")

        file_path = f"songs/{s}.mp3"
        if os.path.exists(file_path):
            audio_file = open(file_path, "rb")
            st.audio(audio_file.read(), format="audio/mp3")
        else:
            st.error(f"File not found: {file_path}")
