import os
import streamlit as st
from surprise import *
from surprise.model_selection import *

st.markdown('# 前 K 個最相似的影評者')
st.markdown('## (User-based) (KNN)')

def get_user_id():
    file_name = 'ml-100k/u.user'
    uid_to_name = {}
    name_to_uid = {}
    with open(file_name) as f:
        for line in f:
            line = line.split("|")
            uid_to_name[line[0]] = line[1]
            name_to_uid[line[1]] = line[0]
    return uid_to_name, name_to_uid

raw_id_to_name, name_to_raw_id = get_user_id()
knn = st.selectbox('選擇方法', ['KNNBasic', 'KNNWithMeans', 'KNNWithZScore', 'KNNBaseline'])
sim = st.selectbox('相似度計算方法', ['cosine', 'msd', 'pearson', 'pearson_baseline'])
user_name = st.selectbox('選擇影評者', list(raw_id_to_name.values()))
number = st.text_input('輸入前幾個', '10')

os.environ['SURPRISE_DATA_FOLDER'] = 'ml-100k'
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

def get_data():
    files_dir = 'ml-100k/'
    reader = Reader("ml-100k")

    train_file = files_dir + "u%d.base"
    test_file = files_dir + "u%d.test"
    folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()
    train_data = pkf.split(data)
    return train_data

if knn == 'KNNBasic':
    algo = KNNBasic(sim_options={'name': sim, 'user_based': False})
elif knn == 'KNNWithMeans':
    algo = KNNWithMeans(sim_options={'name': sim, 'user_based': False})
elif knn == 'KNNWithZScore':
    algo = KNNWithZScore(sim_options={'name': sim, 'user_based': False})
elif knn == 'KNNBaseline':
    algo = KNNBaseline(sim_options={'name': sim, 'user_based': False})

data = get_data()
for trainset, testset in data:
    algo.fit(trainset)

user_id = name_to_raw_id[user_name]
user_inner_id = algo.trainset.to_inner_iid(user_id)
user_neighbors = algo.get_neighbors(user_inner_id, k=int(number))

st.write('前', number, '名和', 'No.', user_name, '最相似的影評者')
for user_inner_id in user_neighbors:
    user_raw_id = algo.trainset.to_raw_iid(user_inner_id)
    user_name = raw_id_to_name[user_raw_id]
    st.write('No.', user_name)
