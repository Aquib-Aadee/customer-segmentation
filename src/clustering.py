from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def perform_clustering(scaled_features, num_clusters=3):
    """
    Applies K-Means clustering to the scaled data.
    """
    print(f"Applying K-Means to find {num_clusters} clusters...")
    
    # Initialize the K-Means model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    
    # Fit the model and get the cluster labels (Group 0, Group 1, or Group 2)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Evaluate the clusters using the Silhouette Score (1 is perfect, -1 is terrible)
    score = silhouette_score(scaled_features, cluster_labels)
    print(f"Model Evaluation (Silhouette Score): {score:.2f}")
    
    return kmeans, cluster_labels

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Clustering model saved as {filename}")