import streamlit as st
import pandas as pd
from surprise import *
from surprise.model_selection import *
from surprise import accuracy

st.markdown('# 前 K 個最相似的影評者')
st.markdown('## (User-based) (KNN)')

knn = st.selectbox('選擇方法', ['KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'KNNBaseline'], index=3)
sim = st.selectbox('相似度計算方法', ['cosine', 'msd', 'pearson', 'pearson_baseline'], index=3)
cv_bool = st.checkbox('交叉驗證')
if cv_bool:
    split_num = st.text_input('輸入分割數', '3')
user_id = st.text_input('選擇影評者 (1 ~ 943)', '24')
number = st.text_input('輸入前幾個', '10')

if knn == 'KNNBasic':
    algo = KNNBasic(sim_options={'name': sim})
elif knn == 'KNNWithMeans':
    algo = KNNWithMeans(sim_options={'name': sim})
elif knn == 'KNNWithZScore':
    algo = KNNWithZScore(sim_options={'name': sim})
elif knn == 'KNNBaseline':
    algo = KNNBaseline(sim_options={'name': sim})

data = Dataset.load_builtin('ml-100k')

if cv_bool:
    cv = KFold(n_splits=int(split_num))
    i = 1
    for trainset, testset in cv.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        st.write(f'第 {i} 次, RMSE : {accuracy.rmse(predictions, verbose=True):.2f} MAE : {accuracy.mae(predictions, verbose=True):.2f}')
        i += 1
else:
    trainset = data.build_full_trainset()
    algo.fit(trainset)

user_inner_id = algo.trainset.to_inner_iid(user_id)
user_neighbors = algo.get_neighbors(user_inner_id, k=int(number))

path = 'ml-100k/u.user'
header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
df = pd.read_csv(path, sep='|', names=header)

st.markdown(f'### 前{number}名和 No.{user_id} 最相似的影評者')
df_copy = df.copy()
df_copy = df_copy[df_copy['user_id'] == int(user_id)]
st.write(df_copy)

user_ids = []

for user_inner_id in user_neighbors:
    user_raw_id = algo.trainset.to_raw_iid(user_inner_id)
    user_ids.append(int(user_raw_id))

df = df[df['user_id'].isin(user_ids)]
st.write(df)
