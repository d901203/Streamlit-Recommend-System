import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pgeocode
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.ticker import MaxNLocator

# Display the title
st.title("Users")

# Load the data
path = "ml-100k/u.user"
header = ["user_id", "age", "gender", "occupation", "zip_code"]
user = pd.read_csv(path, sep="|", names=header)

path = "ml-100k/u.occupation"
jobs = [j[0] for j in pd.read_csv(path, header=None).values]

path = "ml-100k/u.data"
header = ["user_id", "item_id", "rating", "timestamp"]
rating = pd.read_csv(path, sep="\t", names=header)
rating["timestamp"] = pd.to_datetime(rating["timestamp"], unit="s")
year = set(rating["timestamp"].dt.year)
year = list(sorted(year))
user["avg_rating"] = rating.groupby("user_id")["rating"].mean().reset_index(drop=True)
user["rating_count"] = rating.groupby("user_id")["rating"].count().reset_index(drop=True)

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
movie = pd.read_csv(path, sep="|", names=header, encoding="latin-1")
movie["release_date"] = pd.to_datetime(movie["release_date"])

# Display the data
tabs = st.tabs(["Gender", "Age", "Occupation", "Occupation average rating", "Region"])
with tabs[0]:
    male = len(user[user["gender"] == "M"].value_counts())
    female = len(user[user["gender"] == "F"].value_counts())
    labels = ["Male", "Female"]
    values = [male, female]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    st.plotly_chart(fig)

with tabs[1]:
    user.loc[user["gender"] == "M", "gender"] = "Male"
    user.loc[user["gender"] == "F", "gender"] = "Female"
    fig = px.histogram(user, x="age", color="gender", text_auto=True)
    st.plotly_chart(fig)

with tabs[2]:
    path = "ml-100k/u.occupation"
    header = ["occupation"]
    occupation = pd.read_csv(path, sep="|", names=header)
    male = [0] * len(occupation)
    female = [0] * len(occupation)
    jobs = []
    for i, row in occupation.iterrows():
        jobs.append(row["occupation"])
        male[i] = len(user[(user["occupation"] == row["occupation"]) & (user["gender"] == "Male")].value_counts())
        female[i] = len(user[(user["occupation"] == row["occupation"]) & (user["gender"] == "Female")].value_counts())
    fig = go.Figure(
        data=[
            go.Bar(name="Male", x=jobs, y=male, text=male, textposition="auto"),
            go.Bar(
                name="Female",
                x=jobs,
                y=female,
                text=female,
                textposition="auto",
            ),
        ]
    )
    fig.update_xaxes(tickangle=315)
    fig.update_layout(
        xaxis_title="Occupation",
        yaxis_title="Count",
    )
    st.plotly_chart(fig)

with tabs[3]:
    m = {}
    for j in jobs:
        m[j] = user[user["occupation"] == j]["avg_rating"].mean()
    jobs_avg_rating = [round(m[j], 2) for j in jobs]
    fig = go.Figure(
        data=go.Bar(
            x=jobs,
            y=jobs_avg_rating,
            text=jobs_avg_rating,
            textposition="auto",
        )
    )
    fig.update_xaxes(tickangle=315)
    fig.update_layout(
        xaxis_title="Occupation",
        yaxis_title="Average rating",
    )
    st.plotly_chart(fig)


@st.cache_data
def get_address():
    nomi = pgeocode.Nominatim("us")
    addr = [] * len(user)
    for _, row in user.iterrows():
        location = nomi.query_postal_code(row["zip_code"])
        if not pd.isnull(location["latitude"]) and not pd.isnull(location["longitude"]):
            addr.append((location.latitude, location.longitude))
    return addr


with tabs[4]:
    addr = get_address()
    df = pd.DataFrame(addr, columns=["lat", "lon"])
    fig = go.Figure(
        data=go.Scattergeo(
            locationmode="USA-states",
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
        )
    )
    fig.update_layout(
        geo_scope="usa",
    )
    st.plotly_chart(fig)

age_option = st.slider("Age", 7, 73, (7, 15))
st.write("Age range", age_option[0], " ~ ", age_option[1])
gender_option = st.multiselect("Gender", ["Male", "Female"])
job_option = st.multiselect("Occupation", jobs)

user = user[(user["age"] >= age_option[0]) & (user["age"] <= age_option[1])]

if gender_option == ["Male"]:
    user = user[user["gender"] == "Male"]
elif gender_option == ["Female"]:
    user = user[user["gender"] == "Female"]

if job_option:
    user = user[user["occupation"].isin(job_option)]

st.write("Found", len(user), "people")

sort_option = st.selectbox(
    "Sort by",
    [
        "Age (young → old)",
        "Age (old → young)",
        "Rating (high → low)",
        "Rating (low → high)",
        "Number of Ratings (many to few)",
        "Number of Ratings (few to many)",
    ],
    index=0,
)

if sort_option == "Age (young → old)":
    user = user.sort_values(by=["age"], ascending=True)
elif sort_option == "Age (old → young)":
    user = user.sort_values(by=["age"], ascending=False)
elif sort_option == "Rating (high → low)":
    user = user.sort_values(by=["avg_rating"], ascending=False)
elif sort_option == "Rating (low → high)":
    user = user.sort_values(by=["avg_rating"], ascending=True)
elif sort_option == "Number of Ratings (many to few)":
    user = user.sort_values(by=["rating_count"], ascending=False)
elif sort_option == "Number of Ratings (few to many)":
    user = user.sort_values(by=["rating_count"], ascending=True)

for _, row in user.iterrows():
    container = st.container()
    container.title(f'No. {row["user_id"]}')
    container.markdown(f'Age : {row["age"]}')
    container.markdown(f'Gender : {row["gender"]}')
    container.markdown(f'Occupation : {row["occupation"]}')
    container.markdown(f'Zip code : {row["zip_code"]}')
    container.markdown(f'Average rating : {row["avg_rating"]:.2f}')
    container.markdown(f'Number of ratings : {row["rating_count"]}')
    expander = container.expander("More")
    tabs = expander.tabs(
        [
            "Rating distribution",
            "Average rating per year",
            "Number of ratings per year",
            "Genre of ratings statistics",
        ]
    )
    with tabs[0]:
        r = rating.copy()
        r = rating[rating["user_id"] == row["user_id"]]["rating"].tolist()
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
        avg_rating = [0] * len(year)
        for i, y in enumerate(year):
            val = rating[(rating["user_id"] == row["user_id"]) & (rating["timestamp"].dt.year == y)]["rating"].mean()
            avg_rating[i] = val if not pd.isnull(val) else 0

        fig, ax = plt.subplots()
        ax.set_xlabel("Year")
        ax.set_ylabel("Average rating")
        ax.set_xticks(year)
        ax.set_ylim(1, 5)
        ax.set_yticks(np.arange(0, 5.2, 0.2))

        ax.plot(year, avg_rating, "go-", label="rating")
        ax.legend()

        for i, v in enumerate(avg_rating):
            ax.annotate(
                f"{v:0.2f}",
                (year[i], v),
                xytext=(year[i], v + 0.1),
                ha="center",
            )
        st.pyplot(fig)

    with tabs[2]:
        rating_count = [0] * len(year)
        for i, y in enumerate(year):
            val = len(rating[(rating["user_id"] == row["user_id"]) & (rating["timestamp"].dt.year == y)])
            rating_count[i] = val if not pd.isnull(val) else 0

        fig, ax = plt.subplots()
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of ratings")
        ax.set_xticks(year)
        mx = max(rating_count)
        ax.set_ylim(top=(mx // 10 + 2) * 10)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        p = ax.bar(year, rating_count)
        ax.bar_label(p)
        st.pyplot(fig)

    with tabs[3]:
        df = pd.merge(rating[rating["user_id"] == row["user_id"]], movie, on="item_id")
        genre_copy = genre.copy()
        genre_count = [0] * len(genre_copy)
        for i, g in enumerate(genre_copy):
            genre_count[i] = len(df[df[g] == 1])
        genre_copy = [genre_copy[i] for i in range(len(genre_copy)) if genre_count[i] != 0]
        genre_count = [i for i in genre_count if i != 0]
        fig = go.Figure(data=[go.Pie(labels=genre_copy, values=genre_count)])
        st.plotly_chart(fig)
