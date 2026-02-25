# Customer Segmentation: An Unsupervised Learning Analysis

## Project Motivation
This repository explores unsupervised machine learning techniques to identify latent behavioral structures within unlabeled consumer data. It was developed to demonstrate proficiency in clustering algorithms, mathematical feature scaling, and distance-based metric evaluation as part of my academic preparation for graduate-level studies in Data Science.

## Methodology
The objective is to partition a customer base into distinct, non-overlapping subgroups based on purchasing behavior and annual income. The pipeline executes the following workflow:
1. **Feature Standardization:** Utilizing `StandardScaler` to normalize variance. This is mathematically critical for distance-based algorithms (which rely on Euclidean distance) to prevent features with larger magnitudes from dominating the objective function.
2. **Clustering Algorithm:** Implementing **K-Means Clustering**, an iterative algorithm that partitions the data into $k=3$ clusters by minimizing the within-cluster sum of squares (WCSS).
3. **Evaluation:** Assessing cluster cohesion and separation using the **Silhouette Coefficient**, an intrinsic metric that validates the clustering configuration without requiring external ground-truth labels.

## Technical Implementation
* **Language:** Python
* **Core Libraries:** `pandas`, `scikit-learn`, `joblib`
* **Architecture:** Modular script design separating data preprocessing (`src/data_preprocessing.py`) from model initialization and application (`src/clustering.py`).

## Future Work & Limitations
To expand upon this foundational analysis, future iterations will explore:
* Implementing the **Elbow Method** to systematically and mathematically justify the optimal selection of the hyperparameter $k$.
* Comparing centroid-based K-Means against density-based algorithms like **DBSCAN** to better handle potential non-globular cluster shapes and statistical outliers.
* Integrating **Principal Component Analysis (PCA)** to reduce dimensionality and visualize decision boundaries if additional features are introduced.

## Replication Instructions
To execute this pipeline locally and generate the clustered dataset:
```bash
git clone [https://github.com/Aquib-Aadee/customer-segmentation.git](https://github.com/Aquib-Aadee/customer-segmentation.git)
cd customer-segmentation
pip install -r requirements.txt
python main.py