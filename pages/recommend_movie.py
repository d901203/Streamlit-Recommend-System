import streamlit as st
import pandas as pd
from surprise import *
from surprise.model_selection import *

st.title('電影推薦 (SVD)')
st.info('1. 利用 SVD 先對 u.data 進行訓練\n2. 生成 u.data 裡面沒有的 anti_data 進行評分預測\n3. 依照評分預測結果推薦電影給使用者\n')

def get_top_n(predictions, user_id, n=10):
    top_n = []
    for uid, iid, _, est, _ in predictions:
        if uid == user_id:
            top_n.append((iid, est))
    top_n.sort(key=lambda x: x[1], reverse=True)
    return top_n[:n]

def get_data():
    data = Dataset.load_builtin("ml-100k")
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    return predictions

user_id = st.text_input('請輸入使用者 ID (1 ~ 943)', '196')
num_movies = st.text_input('請輸入推薦電影數量 (1 ~ 1682)', '10')

predictions = get_data()
top_n = get_top_n(predictions, user_id, n=int(num_movies))


path = 'ml-100k/u.genre'
genre = [g[0] for g in pd.read_csv(path, sep='|', header=None).values]

path = 'ml-100k/u.item'
header = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + genre
df = pd.read_csv(path, sep='|', names=header, encoding='latin-1')
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_date'] = df['release_date'].dt.date

st.write('為使用者', user_id, '推薦的', num_movies, '部電影為 : ')
movie_name = []
realease_year = []
pred_rating = []
for movie_id, rating in top_n:
    movie_name.append(df[df['item_id'] == int(movie_id)]['movie_title'].values[0])
    realease_year.append(df[df['item_id'] == int(movie_id)]['release_date'].values[0])
    pred_rating.append(rating)
df = pd.DataFrame({'電影名稱': movie_name, '發行日期': realease_year, '預測評分': pred_rating})
st.write(df)



