import streamlit as st
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_pickle('data.pkl')

similarity = pickle.load(open('similarity.pkl','rb'))
nlp = pickle.load(open('nlp_model.pkl','rb'))

def get_posters(movie_name):
    id = df[df['title'] == movie_name].id.iloc[0]
    url = "https://api.themoviedb.org/3/movie/{}?api_key=f0f5c744c77c2bcdb0188e44992ba573&language=en-US".format(id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path

    return full_path

def get_reviews(movie_name):
    movie_id = df[df['title'] == movie_name].id.iloc[0]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?language=en-US&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmMGY1Yzc0NGM3N2MyYmNkYjAxODhlNDQ5OTJiYTU3MyIsInN1YiI6IjY0YTlhMjI4M2UyZWM4MDBjYmNkNGQ4YyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.WXTLJov3SE_k4eMqFmIeSDBwj3XpPePWznFDrE2fIAs"
    }
    response = requests.get(url, headers=headers)
    if len(pd.DataFrame(response.json())) == 0:
        return 'No Reviews'
    else:
        return list(pd.DataFrame(response.json()['results'])['content'])



def recommend(movie):
    index = df[df['title'] == movie].index[0]
    similar_movies = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)[1:6]
    names = []
    posters = []
    for i in similar_movies:
        names.append(df.iloc[i[0]].title)
        posters.append(get_posters(df.iloc[i[0]].title))
    
    return names, posters

nltk.download("stopwords")
stopset = stopwords.words('english')
vectorizer = TfidfVectorizer(use_idf = True,lowercase = True, strip_accents='ascii',stop_words=stopset)

st.title('Movie Recommender System')
selected_movie = st.selectbox('Select a Movie', df['title'])

st.write('You selected:', selected_movie)

if st.button('Recommend'):
    st.write('Here are some recommendations:')

    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    for i in range(len(recommended_movie_names)):
        col1, col2 = st.columns(2)

        with col1:
            # Adjust the width parameter to control image size
            st.image(recommended_movie_posters[i], width=200)  # Adjust the width value as needed

        with col2:
            st.write(f'**{recommended_movie_names[i]}**')
            table_data = [
                    [f'{get_reviews(recommended_movie_names[i])[0][:150]}', f"{nlp.predict(vectorizer.transform(get_reviews(recommended_movie_names[i])[0]))}"],
                    [f'{get_reviews(recommended_movie_names[i])[1][:150]}', f"{nlp.predict(vectorizer.transform(get_reviews(recommended_movie_names[i])[1]))}"]
                ]
            st.table(table_data)


