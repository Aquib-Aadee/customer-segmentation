from src.data_preprocessing import load_and_scale_data
from src.clustering import perform_clustering, save_model

if __name__ == "__main__":
    print("Starting Customer Segmentation Pipeline...")
    
    # 1. Load and scale the data
    scaled_data, original_df = load_and_scale_data('data/raw_data.csv')
    
    # 2. Perform K-Means Clustering
    model, labels = perform_clustering(scaled_data, num_clusters=3)
    
    # 3. Attach the predicted group labels back to our original data
    original_df['Customer_Group'] = labels
    
    # 4. Save the results and the model
    original_df.to_csv('data/clustered_customers.csv', index=False)
    save_model(model, 'models/kmeans_model.pkl')
    
    print("Pipeline completed! Clustered data saved to data/clustered_customers.csv")