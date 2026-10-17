
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# ----------------------------------------
# 1. Paths and data
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

CUSTOMER_FILE = (
    BASE_DIR / "reports" / "eda" / "customer_segments.csv"
)

OUTPUT_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(CUSTOMER_FILE, index_col=0)

# Use the same customer features as K-Means.
feature_columns = [
    "purchase_records",
    "total_spending",
    "total_units",
    "average_order_value",
    "category_diversity",
]

X = customers[feature_columns]

# Standardize features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# 2. Evaluate different cluster counts
# ----------------------------------------

results = []

for k in range(2, 7):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "clusters": k,
        "silhouette_score": score,
    })

results_df = pd.DataFrame(results)

print("\nHIERARCHICAL CLUSTER EVALUATION")
print(results_df.to_string(index=False))

best_k = int(
    results_df.loc[
        results_df["silhouette_score"].idxmax(),
        "clusters",
    ]
)

print("\nBest tested cluster count:", best_k)

# ----------------------------------------
# 3. Train the final model
# ----------------------------------------

final_model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward",
)

customers["hierarchical_cluster"] = (
    final_model.fit_predict(X_scaled)
)

# ----------------------------------------
# 4. Cluster summary
# ----------------------------------------

summary = customers.groupby(
    "hierarchical_cluster"
).agg(
    customer_count=("purchase_records", "size"),
    avg_purchase_records=("purchase_records", "mean"),
    avg_spending=("total_spending", "mean"),
    avg_units=("total_units", "mean"),
    avg_order_value=("average_order_value", "mean"),
)

print("\nHIERARCHICAL CLUSTER SUMMARY")
print(summary.round(2).to_string())

customers.to_csv(
    REPORT_DIR / "hierarchical_customer_segments.csv"
)

summary.to_csv(
    REPORT_DIR / "hierarchical_cluster_summary.csv"
)

# ----------------------------------------
# 5. Create dendrogram
# ----------------------------------------

Z = linkage(X_scaled, method="ward")

plt.figure(figsize=(14, 6))

dendrogram(
    Z,
    no_labels=True,
    truncate_mode="lastp",
    p=30,
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer groups")
plt.ylabel("Ward distance")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "customer_dendrogram.png",
    dpi=150,
)

plt.close()

# ----------------------------------------
# 6. Save reusable preprocessing object
# ----------------------------------------

joblib.dump(
    scaler,
    OUTPUT_DIR / "hierarchical_scaler.joblib",
)

print("\nHierarchical clustering completed.")
print("Results saved in:", REPORT_DIR)
