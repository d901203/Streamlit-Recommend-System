import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
from collections import Counter

st.title("Movies")

path = "ml-100k/u.genre"
genre = [g[0] for g in pd.read_csv(path, sep="|", header=None).values]

path = "ml-100k/u.user"
header = ["user_id", "age", "gender", "occupation", "zip_code"]
user = pd.read_csv(path, sep="|", names=header)

path = "ml-100k/u.data"
header = ["user_id", "item_id", "rating", "timestamp"]
rating = pd.read_csv(path, sep="\t", names=header)
rating = rating.merge(user, on="user_id")
rating["timestamp"] = pd.to_datetime(rating["timestamp"], unit="s")
avg_rating = rating.groupby("item_id")["rating"].mean().reset_index(drop=True)

path = "ml-100k/u.item"
header = [
    "item_id",
    "movie_title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
] + genre
movie = pd.read_csv(path, sep="|", names=header, encoding="latin-1")
movie["release_date"] = pd.to_datetime(movie["release_date"])
movie["avg_rating"] = avg_rating
movie["rating_count"] = (
    rating.groupby("item_id")["rating"].count().reset_index(drop=True)
)

tabs = st.tabs(
    [
        "Number of movies released per year",
        "Rating",
        "Number of ratings",
        "Genre",
        "Genre average rating",
    ]
)
with tabs[0]:
    year = set(movie["release_date"].dt.year)
    year = list(sorted(year))
    y = [0] * len(year)
    for i, x in enumerate(year):
        y[i] = len(movie[movie["release_date"].dt.year == x])
    fig = go.Figure(data=[go.Bar(x=year, y=y, text=y, textposition="auto")])
    fig.update_layout(xaxis_title="Year", yaxis_title="Number of movies")
    st.plotly_chart(fig)

with tabs[1]:
    genre_count = [0] * len(genre)
    for i, g in enumerate(genre):
        genre_count[i] = len(movie[movie[g] == 1])
    fig = go.Figure(data=[go.Pie(labels=genre, values=genre_count)])
    st.plotly_chart(fig)

with tabs[2]:
    fig = go.Figure(
        data=[go.Histogram(x=movie["avg_rating"], texttemplate="%{y}")]
    )
    fig.update_layout(xaxis_title="Rating", yaxis_title="Number of movies")
    st.plotly_chart(fig)

with tabs[3]:
    fig = go.Figure(
        data=[go.Histogram(x=movie["rating_count"], texttemplate="%{y}")]
    )
    fig.update_layout(
        xaxis_title="Rating count", yaxis_title="Number of movies"
    )
    st.plotly_chart(fig)

with tabs[4]:
    genre_copy = genre.copy()
    m = {}
    for g in genre_copy:
        m[g] = movie[movie[g] == 1]["avg_rating"].mean()
    genre_avg_rating = [round(m[g], 2) for g in genre]
    fig = go.Figure(
        data=[
            go.Bar(
                x=genre,
                y=genre_avg_rating,
                text=genre_avg_rating,
                textposition="auto",
            )
        ]
    )
    fig.update_layout(xaxis_title="Genre", yaxis_title="Average rating")
    st.plotly_chart(fig)

range = st.slider("released year", 1922, 1998, (1922, 1966))
st.write("released year : ", range[0], " ~ ", range[1])

genre_options = st.multiselect("Genre", genre)
rating_options = st.slider("Rating", 1, 5, (2, 3))
st.write(
    "Rating range : ", rating_options[0], "point ~ ", rating_options[1], "point"
)

for g in genre_options:
    movie = movie[movie[g] == 1]
movie = movie[
    (movie["release_date"].isnull())
    | (
        (movie["release_date"].dt.year >= range[0])
        & (movie["release_date"].dt.year <= range[1])
    )
]
movie["release_date"] = movie["release_date"].dt.date
movie = movie[
    (movie["avg_rating"] >= rating_options[0])
    & (movie["avg_rating"] <= rating_options[1])
]
st.write("Found ", len(movie), " movies")

option = st.selectbox(
    "Sort by",
    [
        "Released Year (old → new)",
        "Released Year (new → old)",
        "Rating (high → low)",
        "Rating (low → high)",
        "Number of Ratings (many to few)",
        "Number of Ratings (few to many)",
    ],
    index=0,
)

if option == "Released Year (old → new)":
    movie = movie.sort_values(by="release_date", ascending=True)
elif option == "Released Year (new → old)":
    movie = movie.sort_values(by="release_date", ascending=False)
elif option == "Rating (high → low)":
    movie = movie.sort_values(by="avg_rating", ascending=False)
elif option == "Rating (low → high)":
    movie = movie.sort_values(by="avg_rating", ascending=True)
elif option == "Number of Ratings (many to few)":
    movie = movie.sort_values(by="rating_count", ascending=False)
elif option == "Number of Ratings (few to many)":
    movie = movie.sort_values(by="rating_count", ascending=True)

for i, row in movie.iterrows():
    container = st.container()
    container.title(f"{row['movie_title']}")
    container.markdown(f"Released date : {row['release_date']}")
    container.markdown(
        f"Genre : {', '.join([g for g in genre if row[g] == 1])}"
    )
    container.markdown(
        f"Average rating : {row['avg_rating']:0.2f} (Number of ratings :"
        f" {row['rating_count']})"
    )
    expander = container.expander("More")
    tabs = expander.tabs(
        [
            "Rating distribution",
            "Average rating per year",
            "Number of ratings per year",
            "Occupation of ratings statistics",
        ]
    )
    with tabs[0]:
        r = rating.copy()
        r = rating[rating["item_id"] == row["item_id"]]["rating"].tolist()
        r = [r.count(i) for i in [1, 2, 3, 4, 5]]
        fig, ax = plt.subplots()
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of ratings")
        ax.set_xticks(np.arange(1, 6, 1))
        ax.set_ylim(top=(max(r) // 10 + 2) * 10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        p = ax.bar(np.arange(1, 6, 1), r)
        ax.bar_label(p)
        st.pyplot(fig)

    with tabs[1]:
        year = set(rating["timestamp"].dt.year)
        year = list(sorted(year))
        y = [0] * len(year)
        for i, x in enumerate(year):
            val = rating[
                (rating["item_id"] == row["item_id"])
                & (rating["timestamp"].dt.year == x)
            ]["rating"].mean()
            y[i] = val if not np.isnan(val) else 0

        fig, ax = plt.subplots()
        ax.set_xlabel("Year")
        ax.set_ylabel("Average rating")
        ax.set_xticks(year)
        ax.set_ylim(1, 5)
        ax.set_yticks(np.arange(0, 5.2, 0.2))

        ax.plot(year, y, "go-", label="rating")
        ax.legend()

        for i, v in enumerate(y):
            ax.annotate(
                f"{v:0.2f}",
                (year[i], v),
                xytext=(year[i], v + 0.1),
                ha="center",
            )
        st.pyplot(fig)

    with tabs[2]:
        year = set(rating["timestamp"].dt.year)
        year = list(sorted(year))
        y2 = [0] * len(year)
        y3 = [0] * len(year)
        for i, x in enumerate(year):
            y2[i] = len(
                rating[
                    (rating["item_id"] == row["item_id"])
                    & (rating["gender"] == "M")
                    & (rating["timestamp"].dt.year == x)
                ]
            )
            y3[i] = len(
                rating[
                    (rating["item_id"] == row["item_id"])
                    & (rating["gender"] == "F")
                    & (rating["timestamp"].dt.year == x)
                ]
            )

        fig, ax = plt.subplots()
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of ratings")
        ax.set_xticks(year)
        mx = max([i + j for i, j in zip(y2, y3)])
        ax.set_ylim(top=(mx // 10 + 2) * 10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        p1 = ax.bar(year, y2, label="Male")
        p2 = ax.bar(year, y3, bottom=y2, label="Female")
        ax.legend()

        for c in ax.containers:
            labels = [v if v != 0 else "" for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type="center")
        ax.bar_label(p2)
        st.pyplot(fig)

    with tabs[3]:
        df = rating[rating["item_id"] == row["item_id"]]
        jobs = df["occupation"].tolist()
        cnt = Counter(jobs)
        jobs = list(set(jobs))
        jobs_cnt = [cnt[j] for j in jobs]
        fig = go.Figure(data=[go.Pie(labels=jobs, values=jobs_cnt)])
        st.plotly_chart(fig)
