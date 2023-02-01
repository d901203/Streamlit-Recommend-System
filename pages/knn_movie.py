import streamlit as st
import pandas as pd
from surprise import *
from surprise.model_selection import *
from surprise import accuracy

st.markdown("# Top K Most Similar Movies")
st.markdown("## (Item-base) (KNN)")


def get_movie_id():
    file_name = "ml-100k/u.item"
    rid_to_name = {}
    name_to_rid = {}
    with open(file_name, encoding="latin-1") as f:
        for line in f:
            line = line.split("|")
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid


raw_id_to_name, name_to_raw_id = get_movie_id()
knn = st.selectbox(
    "KNN Algorithms",
    ["KNNBasic", "KNNWithMeans", "KNNWithZScore", "KNNBaseline"],
    index=3,
)
sim = st.selectbox(
    "Similarity Calculation Algorithms",
    ["cosine", "msd", "pearson", "pearson_baseline"],
    index=3,
)
cv_bool = st.checkbox("Enable Cross-validation")
if cv_bool:
    split_num = st.text_input("Number of subset", "3")
movie_name = st.selectbox(
    "Input the movie name", list(raw_id_to_name.values())
)
number = st.text_input("Top K", "10")

if knn == "KNNBasic":
    algo = KNNBasic(sim_options={"name": sim, "user_based": False})
elif knn == "KNNWithMeans":
    algo = KNNWithMeans(sim_options={"name": sim, "user_based": False})
elif knn == "KNNWithZScore":
    algo = KNNWithZScore(sim_options={"name": sim, "user_based": False})
elif knn == "KNNBaseline":
    algo = KNNBaseline(sim_options={"name": sim, "user_based": False})

data = Dataset.load_builtin("ml-100k", prompt=False)

if cv_bool:
    cv = KFold(n_splits=int(split_num))
    i = 1
    for trainset, testset in cv.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        st.write(
            f"{i} : RMSE : "
            f" {accuracy.rmse(predictions, verbose=True):.2f} MAE : "
            f" {accuracy.mae(predictions, verbose=True):.2f}"
        )
        i += 1
else:
    trainset = data.build_full_trainset()
    algo.fit(trainset)

movie_id = name_to_raw_id[movie_name]
movie_inner_id = algo.trainset.to_inner_iid(movie_id)
movie_neighbors = algo.get_neighbors(movie_inner_id, k=int(number))

st.markdown(f"### TOP {number} Most Similar Movies to {movie_name}")
movie_names = []
for movie_inner_id in movie_neighbors:
    movie_raw_id = algo.trainset.to_raw_iid(movie_inner_id)
    mn = raw_id_to_name[movie_raw_id]
    movie_names.append(mn)

path = "ml-100k/u.genre"
genre = [g[0] for g in pd.read_csv(path, sep="|", header=None).values]

path = "ml-100k/u.item"
header = [
    "item_id",
    "movie_title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
] + genre
df = pd.read_csv(path, sep="|", names=header, encoding="latin-1")
df["release_date"] = pd.to_datetime(df["release_date"])
df["release_date"] = df["release_date"].dt.date

df_copy = df.copy()
df_copy = df_copy[df_copy["movie_title"] == movie_name]
df_copy = df_copy[["movie_title", "release_date"] + genre]
st.write(df_copy)

df = df[df["movie_title"].isin(movie_names)]
df = df[["movie_title", "release_date"] + genre]
st.write(df)
