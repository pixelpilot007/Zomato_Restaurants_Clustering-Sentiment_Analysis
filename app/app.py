# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Zomato Sentiment & Restaurant Clustering",
    page_icon="🍴",
    layout="wide"
)

# Apply seaborn theme
sns.set_theme(style="whitegrid")

# -------------------------
# 1. Load Data
# -------------------------
import os

  # directory of app.py
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    restaurants = pd.read_csv(os.path.join(BASE_DIR, "Restaurant names and Metadata.csv"))
    reviews = pd.read_csv(os.path.join(BASE_DIR, "Zomato Restaurant reviews.csv"))
    return restaurants, reviews

restaurants, reviews = load_data()

# -------------------------
# 2. App Title & Sidebar
# -------------------------
st.title("🍴 Zomato Sentiment & Restaurant Clustering App")
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Data Wrangling", "Visualization", "Sentiment Analysis", "Clustering"])

# -------------------------
# 3. Dataset Overview
# -------------------------
if page == "Dataset Overview":
    st.header("📂 Dataset Overview")
    st.subheader("Restaurant Metadata")
    st.dataframe(restaurants.head(10), use_container_width=True)
    st.subheader("Restaurant Reviews")
    st.dataframe(reviews.head(10), use_container_width=True)
    st.success(f"✅ Restaurants: {restaurants.shape}, Reviews: {reviews.shape}")

# -------------------------
# 4. Data Wrangling
# -------------------------
elif page == "Data Wrangling":
    st.header("🧹 Data Wrangling")
    restaurants['Cost'] = restaurants['Cost'].astype(str).str.replace(',', '').astype(float)

    reviews['Review'] = reviews['Review'].replace("Like", np.nan)
    reviews.dropna(subset=['Review'], inplace=True)

    st.info("Cleaned Restaurant `Cost` & `Review` Columns")
    st.dataframe(restaurants[['Name', 'Cost']].head(10), use_container_width=True)
    st.dataframe(reviews[['Review']].head(10), use_container_width=True)

# -------------------------
# 5. Visualization
# -------------------------
elif page == "Visualization":
    st.header("📊 Visualization")

    # WordCloud: Expensive vs Cheap restaurants
    restaurants['Cost'] = restaurants['Cost'].astype(str).str.replace(',', '', regex=True)
    restaurants['Cost'] = pd.to_numeric(restaurants['Cost'], errors='coerce')
    st.subheader("☁️ WordCloud: Expensive vs Cheap Restaurants")

    expensive = restaurants[restaurants['Cost'] > restaurants['Cost'].median()]
    cheap = restaurants[restaurants['Cost'] <= restaurants['Cost'].median()]

    col1, col2 = st.columns(2)

    with col1:
        st.caption("💰 Expensive Restaurants")
        text_expensive = " ".join(expensive['Cuisines'].dropna())
        if text_expensive.strip():  # only if non-empty
            wc = WordCloud(width=400, height=300, background_color="white", colormap="Reds").generate(text_expensive)
            st.image(np.array(wc.to_image()), use_container_width=True)
        else:
            st.warning("No cuisines found for expensive restaurants.")

    with col2:
        st.caption("💸 Cheap Restaurants")
        text_cheap = " ".join(cheap['Cuisines'].dropna())
        if text_cheap.strip():
            wc = WordCloud(width=400, height=300, background_color="white", colormap="Blues").generate(text_cheap)
            st.image(np.array(wc.to_image()), use_container_width=True)
        else:
            st.warning("No cuisines found for cheap restaurants.")

    # Rating vs Review Length
    st.subheader("⭐ Rating vs Review Length")
    reviews['Review Length'] = reviews['Review'].astype(str).apply(len)

    if 'Rating' in reviews.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=reviews, x="Review Length", y="Rating", ax=ax, alpha=0.5, palette="viridis", hue="Rating", legend=False)
        ax.set_title("Review Length vs Rating")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(reviews['Review Length'], bins=50, ax=ax, color="teal")
        ax.set_title("Distribution of Review Lengths")
        st.pyplot(fig)

# -------------------------
# 6. Sentiment Analysis
# -------------------------
elif page == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis")
    st.write("Using **TextBlob** to analyze polarity of reviews")

    text_input = st.text_area("✍️ Enter a customer review:")
    if text_input:
        polarity = TextBlob(text_input).sentiment.polarity
        sentiment = "Positive 😀" if polarity > 0 else "Negative 😞"
        st.success(f"**Sentiment:** {sentiment} (polarity={polarity:.2f})")

    # Polarity Distribution
    st.subheader("📈 Polarity Distribution")
    reviews['Polarity'] = reviews['Review'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(reviews['Polarity'], bins=20, ax=ax, color="green")
    ax.set_title("Distribution of Review Sentiment Polarity")
    st.pyplot(fig)

# -------------------------
# 7. Clustering
# -------------------------
elif page == "Clustering":
    st.header("🤝 Clustering Restaurants by Cuisines")

    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(restaurants['Cuisines'].dropna())

    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X)

    restaurants['Cluster'] = -1
    restaurants.loc[restaurants['Cuisines'].notna(), 'Cluster'] = clusters

    st.write("🔍 Clustered Restaurants (Top 20)")
    st.dataframe(restaurants[['Name', 'Cuisines', 'Cluster']].head(20), use_container_width=True)

    st.subheader("📊 Cluster Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="Cluster", data=restaurants, ax=ax, palette="Set2")
    ax.set_title("Number of Restaurants in Each Cluster")
    st.pyplot(fig)
