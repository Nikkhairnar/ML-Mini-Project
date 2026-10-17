
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------------------
# 1. Paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "restaurant_sales_cleaned.csv"
)

OUTPUT_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# 2. Load data
# -----------------------------------------

df = pd.read_csv(INPUT_FILE)

# Use complete purchase records only
sales = df[
    df["purchase_details_complete"].astype(str).str.lower()
    .isin(["true", "1"])
].copy()

sales["order_total"] = pd.to_numeric(
    sales["order_total"], errors="coerce"
)
sales["quantity"] = pd.to_numeric(
    sales["quantity"], errors="coerce"
)

sales = sales.dropna(
    subset=[
        "customer_id",
        "order_total",
        "quantity",
        "category",
    ]
)

print("Complete purchase records:", len(sales))
print("Distinct customers:", sales["customer_id"].nunique())

# -----------------------------------------
# 3. Create customer-level features
# -----------------------------------------

customer_features = sales.groupby("customer_id").agg(
    purchase_records=("order_id", "nunique"),
    total_spending=("order_total", "sum"),
    total_units=("quantity", "sum"),
    average_order_value=("order_total", "mean"),
    category_diversity=("category", "nunique"),
)

print("\nCustomer features:")
print(customer_features.head())

# -----------------------------------------
# 4. Standardize features
# -----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(customer_features)

# -----------------------------------------
# 5. Find suitable cluster count
# -----------------------------------------

scores = []

max_k = min(6, len(customer_features) - 1)

for k in range(2, max_k + 1):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    scores.append({
        "k": k,
        "silhouette_score": score,
        "inertia": model.inertia_,
    })

scores_df = pd.DataFrame(scores)

print("\nCluster evaluation:")
print(scores_df.to_string(index=False))

# Choose the highest silhouette score from tested values
best_k = int(
    scores_df.loc[
        scores_df["silhouette_score"].idxmax(), "k"
    ]
)

print("\nSelected number of clusters:", best_k)

# -----------------------------------------
# 6. Train final K-Means model
# -----------------------------------------

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10,
)

customer_features["cluster"] = kmeans.fit_predict(X_scaled)

# Assign interpretable business persona labels based on spending
mean_spending = customer_features.groupby("cluster")["total_spending"].mean()
high_cluster = mean_spending.idxmax()
persona_map = {
    high_cluster: "High-Value Frequent Diners",
    1 - high_cluster: "Standard Value Diners",
}
customer_features["segment_name"] = customer_features["cluster"].map(persona_map)

# -----------------------------------------
# 7. Summarize clusters
# -----------------------------------------

cluster_summary = customer_features.groupby(["cluster", "segment_name"]).agg(
    customer_count=("purchase_records", "size"),
    avg_purchase_records=("purchase_records", "mean"),
    avg_spending=("total_spending", "mean"),
    avg_units=("total_units", "mean"),
    avg_order_value=("average_order_value", "mean"),
    avg_category_diversity=("category_diversity", "mean"),
).reset_index()

print("\nCluster summary:")
print(cluster_summary.round(2).to_string())

# -----------------------------------------
# 8. Save results and model
# -----------------------------------------

customer_features.to_csv(
    REPORT_DIR / "customer_segments.csv"
)

cluster_summary.to_csv(
    REPORT_DIR / "cluster_summary.csv",
    index=False,
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "customer_scaler.joblib"
)

joblib.dump(
    kmeans,
    OUTPUT_DIR / "kmeans_model.joblib"
)

# -----------------------------------------
# 9. Plot silhouette scores
# -----------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    scores_df["k"],
    scores_df["silhouette_score"],
    marker="o",
)

plt.title("K-Means Cluster Evaluation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(scores_df["k"])
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "silhouette_scores.png",
    dpi=150,
)

plt.close()

# -----------------------------------------
# 10. Visualize clusters using PCA
# -----------------------------------------

pca = PCA(n_components=2)
coordinates = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    coordinates[:, 0],
    coordinates[:, 1],
    c=customer_features["cluster"],
    cmap="viridis",
    s=70,
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "customer_clusters.png",
    dpi=150,
)

plt.close()

print("\nK-Means training completed.")
print("Customer segments saved.")
print("Models saved in:", OUTPUT_DIR)
print("Charts saved in:", REPORT_DIR)
