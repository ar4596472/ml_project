import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram

# Load data
data = pd.read_csv("master2.csv")

# Function to perform KMeans clustering
def perform_kmeans(X_scaled, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X_scaled)
    data.loc[:, 'cluster'] = kmeans.labels_
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    return kmeans, silhouette_avg

# Function to print countries in each cluster with categorical labels
def print_countries_by_cluster(data, cluster_labels):
    clusters = {}
    label_mapping = {0: "Underdeveloped Countries", 1: "Developed Countries", 2: "Developing Countries"}
    for label in set(cluster_labels):
        countries = data[cluster_labels == label]['country'].tolist()
        cluster_name = label_mapping[label]
        clusters[cluster_name] = countries
    return clusters

# Function to perform Hierarchical clustering
def perform_hierarchical(X, n_clusters, linkage_method):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    cluster_labels = cluster.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    return cluster, cluster_labels, silhouette_avg

# Streamlit UI
st.title('Clustering Visualization')

# Sidebar options
algorithm = st.sidebar.selectbox("Select Clustering Algorithm", ["KMeans", "Hierarchical"])
if algorithm == "KMeans":
    n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    random_state = st.sidebar.slider("Select Random State", min_value=0, max_value=100, value=42)
elif algorithm == "Hierarchical":
    n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    linkage_method = st.sidebar.selectbox("Select Linkage Method", ["ward", "complete", "average"])

# Select relevant features for clustering
X = data[['suicides/100k pop', 'gdp_per_capita ($)', 'HDI']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if algorithm == "KMeans":
    # Perform KMeans clustering
    kmeans, silhouette_avg_kmeans = perform_kmeans(X_scaled, n_clusters, random_state)

    # Print Silhouette Score for KMeans
    st.write("Silhouette Score (KMeans):", silhouette_avg_kmeans)

    # Print KMeans labels and shape
    st.write("KMeans Labels:", kmeans.labels_)
    st.write("KMeans Labels Shape:", kmeans.labels_.shape)

    # Print countries in each cluster with categorical labels
    cluster_labels_kmeans = kmeans.labels_
    country_clusters_kmeans = print_countries_by_cluster(data, cluster_labels_kmeans)
    st.write("Countries in Each Cluster (KMeans):")
    for cluster, countries in country_clusters_kmeans.items():
        st.write(f"{cluster}: {', '.join(countries)}")

    # Visualize KMeans clustering results
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        ax1.scatter(cluster_data['gdp_per_capita ($)'], cluster_data['suicides/100k pop'], label=f'Cluster {cluster}')
    ax1.set_title('Clusters of Countries based on Suicide Rates and GDP per Capita')
    ax1.set_xlabel('GDP per Capita ($)')
    ax1.set_ylabel('Suicides per 100k Population')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    fig3, ax1 = plt.subplots(figsize=(10, 6))
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        ax1.scatter(cluster_data['gdp_per_capita ($)'], cluster_data['HDI'], label=f'Cluster {cluster}')
    ax1.set_title('Clusters of Countries based on Suicide Rates and GDP per Capita')
    ax1.set_xlabel('GDP per Capita ($)')
    ax1.set_ylabel('HDI')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig3)

    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    for cluster in data['cluster'].unique():
        cluster_data = data[data['cluster'] == cluster]
        ax2.scatter(cluster_data['gdp_per_capita ($)'], cluster_data['HDI'], cluster_data['suicides/100k pop'], label=f'Cluster {cluster}')
    ax2.set_title('Clusters of Countries based on Suicide Rates, GDP per Capita, and HDI')
    ax2.set_xlabel('GDP_per_Capita')
    ax2.set_ylabel('HDI')
    ax2.set_zlabel('Suicides/100k Pop')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

else:
    # Perform Hierarchical clustering
    hierarchical_cluster, cluster_labels_hierarchical, silhouette_avg_hierarchical = perform_hierarchical(X, n_clusters, linkage_method)

    # Print Silhouette Score for Hierarchical clustering
    st.write("Silhouette Score (Hierarchical):", silhouette_avg_hierarchical)

    # Print the cluster labels for Hierarchical clustering
    st.write("Cluster Labels (Hierarchical):")
    st.write(cluster_labels_hierarchical)

    # Print countries in each cluster with categorical labels
    country_clusters_hierarchical = print_countries_by_cluster(data, cluster_labels_hierarchical)
    st.write("Countries in Each Cluster (Hierarchical):")
    for cluster, countries in country_clusters_hierarchical.items():
        st.write(f"{cluster}: {', '.join(countries)}")

    # Visualize the clusters for Hierarchical clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 1], y=X[:, 0], hue=cluster_labels_hierarchical, palette='viridis')
    plt.title('Hierarchical Clustering')
    plt.xlabel('GDP per Capita ($)')
    plt.ylabel('Suicides per 100k Population')
    plt.legend(title='Cluster')
    plt.grid(True)
    st.pyplot(plt)

    # Visualize the clusters for Hierarchical clustering
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X[:, 1], y=X[:, 2], hue=cluster_labels_hierarchical, palette='viridis')
    plt.title('Hierarchical Clustering')
    plt.xlabel('GDP per Capita ($)')
    plt.ylabel('HDI')
    plt.legend(title='Cluster')
    plt.grid(True)
    st.pyplot(plt)


    # Perform hierarchical clustering with linkage method 'ward'
    Z = linkage(X, method=linkage_method)

    # Plot the dendrogram
    plt.figure(figsize=(15, 8))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.,)
    st.pyplot(plt)
