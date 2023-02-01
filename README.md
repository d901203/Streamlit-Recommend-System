# MovieLens Recommend System
This is a movie recommendation system based on the MovieLens dataset. The system uses the Singular Value Decomposition (SVD) algorithm to predict the ratings of unseen movies for each user. The predicted ratings are then used to recommend movies to users.

## Features
* Movie analysis (Data analysis)
* User analysis (Data analysis)
* Top K Most Similar Movies (Item-base) (KNN)
* Top K Most Similar Users (User-based) (KNN)
* Movie Recommendation (SVD)

## Online Demo
You can access a live demo of the app at [Streamlit Cloud](https://movielens-recommend-system.streamlit.app/).

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
1. Clone the repository: `git clone https://github.com/d901203/movielens-recommend-system.git`
2. Navigate to the project directory: `cd movielens-recommend-system`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the app: `streamlit run main.py`

## Installation with Docker
1. Clone the repository: `git clone https://github.com/d901203/movielens-recommend-system.git`
2. Navigate to the project directory: `cd movielens-recommend-system`
3. Build the Docker image: `docker build -t demo .`
4. Run the Docker container: `docker run -p 8501:8501 demo`
5. Access the app in your browser at `http://localhost:8501`