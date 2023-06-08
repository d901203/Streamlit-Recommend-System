import random

import requests
import streamlit as st
from bs4 import BeautifulSoup
from lxml import etree

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
        " Safari/537.36 Edg/114.0.1823.37"
    )
}


# Cache the data so that it doesn't have to be downloaded every time
@st.cache_data
def get_IMDB_top250_links():
    # Get the links to the top 250 movies
    top250_url = "https://www.imdb.com/chart/top/"
    # Get the HTML
    r = requests.get(top250_url, headers=headers)
    # Parse the HTML
    soup = BeautifulSoup(r.text, "lxml")
    # Get the links
    links = [a.attrs.get("href") for a in soup.select("td.titleColumn a")]
    return links


def get_IMDB_top250(link):
    # Get the title, year, rating, and link to the movie
    main_url = "https://www.imdb.com"
    # Get the HTML
    link = main_url + link
    r = requests.get(link, headers=headers)
    # Parse the HTML
    html = etree.HTML(r.text)
    # Get the title, year, rating, and link
    title = html.xpath("/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[2]/div[1]/div/text()")[
        0
    ].split(": ")[1]
    year = html.xpath(
        "/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[2]/div[1]/ul/li[1]/a/text()"
    )[0]
    rating = html.xpath(
        "/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[2]/div[2]/div/div[1]/a/span/div/div[2]/div[1]/span[1]/text()"
    )[0]
    return title, year, rating, link


# Get the links to the top 250 movies
links = get_IMDB_top250_links()
# Display the title
st.markdown("# Random IMDB Top 250")
# Refresh button
st.button("Refresh")
# Get a random movie
r = random.randint(0, 249)
# Get the title, year, rating, and link
title, year, rating, link = get_IMDB_top250(links[r])
# Display the title, year, rating, and link
st.header("Title")
st.markdown(f"## [{title}]({link})")
st.header("Year")
st.subheader(year)
st.header("Rating")
st.subheader(rating)
