import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import plotly.express as px
import pandas as pd
from src.preprocess import load_data, feature_engineering, scale_features
from src.dimensionality import apply_pca
from src.clustering import kmeans_clustering, som_clustering

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation Dashboard")

df = load_data('data/transactions.csv')
df = feature_engineering(df)
scaled = scale_features(df)
pca_data, _ = apply_pca(scaled)

n_clusters = st.sidebar.slider("K-Means Clusters", 2, 10, 4)
k_labels, _ = kmeans_clustering(pca_data, n_clusters=n_clusters)
som_labels, _ = som_clustering(pca_data)

df_viz = pd.DataFrame(pca_data, columns=['PC1','PC2','PC3'])
df_viz['Cluster'] = k_labels.astype(str)

fig = px.scatter_3d(df_viz, x='PC1', y='PC2', z='PC3', color='Cluster', opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

st.write("📊 Data Sample:")
st.dataframe(df.head())
