import streamlit as st

st.markdown("""
# [MovieLens Recommend System](https://github.com/d901203/movielens-recommend-system)
This is a movie recommendation system based on the MovieLens dataset. The system uses the Singular Value Decomposition (SVD) algorithm to predict the ratings of unseen movies for each user. The predicted ratings are then used to recommend movies to users.

## Features
* Movie analysis (Data analysis)
* User analysis (Data analysis)
* Top K Most Similar Movies (Item-base) (KNN)
* Top K Most Similar Users (User-based) (KNN)
* Movie Recommendation (SVD)

## Online Demo
You can access a live demo of the app at [Streamlit Cloud](https://movielens-recommend-system.streamlit.app/)

## Requirements
* python 3.10 or later
* streamlit
* numpy
* pandas
* pgeocode
* matplotlib
* plotly
* scikit_surprise

## Installation

```
git clone https://github.com/d901203/movielens-recommend-system.git
cd movielens-recommend-system
pip install -r requirements.txt
streamlit run main.py
```

with Dockerfile
```
git clone https://github.com/d901203/movielens-recommend-system.git
cd movielens-recommend-system
docker build -t app:latest .
docker run -p 8501:8501 app:latest
```

Access the app in your browser at `http://localhost:8501`
""")
