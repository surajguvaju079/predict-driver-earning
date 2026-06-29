import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
df= pd.read_csv("clusters.csv")
X=df[
[
    "loads_completed",
    "last_login_days",
    "total_spent"
]
]

X_scaled = scalar.fit_transform(X)
model = KMeans(n_clusters=3,
               random_state=42);

model.fit(X_scaled);
df["cluster"]=model.labels_

cluster_stats=(
    df.groupby("cluster").agg(
        avg_loads=("loads_completed","mean"),
        avg_spent=("total_spent","mean"),
        avg_last_login=("last_login_days","mean"),
        users=("user_id","count")
    )
)



print(df)
print(cluster_stats)
