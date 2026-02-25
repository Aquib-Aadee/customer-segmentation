import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_scale_data(filepath):
    """
    Loads customer data and standardizes the features so they are on the same scale.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # We only want to cluster based on Income and Spending Score.
    # We drop 'CustomerID' because it is just an ID, not a mathematical feature.
    features = df[['Annual_Income', 'Spending_Score']]
    
    # StandardScaler transforms the data so it has a mean of 0 and standard deviation of 1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print("Data scaled successfully.")
    
    # Return both the scaled data (for the AI) and the original dataframe (for our records)
    return scaled_features, df