import streamlit as st
import pandas as pd
from surprise import *
from surprise.model_selection import *
from surprise import accuracy

st.markdown("# Top K Most Similar Users")
st.markdown("## (User-based) (KNN)")

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
user_id = st.text_input("Input the user id (1 ~ 943)", "24")
number = st.text_input("Top K", "10")

if knn == "KNNBasic":
    algo = KNNBasic(sim_options={"name": sim})
elif knn == "KNNWithMeans":
    algo = KNNWithMeans(sim_options={"name": sim})
elif knn == "KNNWithZScore":
    algo = KNNWithZScore(sim_options={"name": sim})
elif knn == "KNNBaseline":
    algo = KNNBaseline(sim_options={"name": sim})

data = Dataset.load_builtin("ml-100k")

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

user_inner_id = algo.trainset.to_inner_iid(user_id)
user_neighbors = algo.get_neighbors(user_inner_id, k=int(number))

path = "ml-100k/u.user"
header = ["user_id", "age", "gender", "occupation", "zip_code"]
df = pd.read_csv(path, sep="|", names=header)

st.markdown(f"### TOP {number} Most Similar Users to No.{user_id}")
df_copy = df.copy()
df_copy = df_copy[df_copy["user_id"] == int(user_id)]
st.write(df_copy)

user_ids = []

for user_inner_id in user_neighbors:
    user_raw_id = algo.trainset.to_raw_iid(user_inner_id)
    user_ids.append(int(user_raw_id))

df = df[df["user_id"].isin(user_ids)]
st.write(df)
