import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Data loading
df_train = pd.read_csv("./DATA/train_val_set.csv", sep=";")
df_test  = pd.read_csv("./DATA/test_set.csv",    sep=";")

# Delete very correlated features
# Correlation matrix (upper side)
corr_matrix = df_train.drop(columns=["difference", "Period_ID", "end_perf"]).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Correlation with target
target_corr = df_train.drop(columns=["Period_ID", "end_perf"]).corr()["difference"].abs()

# Columns to drop
to_drop = []
for column in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, column] > 0.9:
            # Keep most correlated with target
            if target_corr[idx] < target_corr[column]:
                to_drop.append(idx)
            else:
                to_drop.append(column)

# Delete double
to_drop = list(set(to_drop))

print(f"Deletion of {len(to_drop)} feature(s) very correlated :", to_drop)

to_drop += ["Period_ID","difference","end_perf"]

# Update dataset

X_train = df_train.drop(columns=to_drop)
y_train = df_train["difference"].values

X_test  = df_test.drop(columns=to_drop)
y_test  = df_test["difference"].values

# HyperParameters
proportions = np.arange(0.05, 1.0, 0.05)   # 5%, 10%, …, 100%
n_repeats   = 20                            # random draw per proportions
rf_params   = {
    "n_estimators": 1000,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 9,
    "random_state": 42,
    "n_jobs": -1
}

results = []

# Loop on proportions
for prop in proportions:
    print(str(int(prop*100))+"%")
    maes = []

    for i in range(n_repeats):
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size = prop,
            random_state=42 + i,
            shuffle=True
        )
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_sub, y_sub)
        y_pred = rf.predict(X_test)
        maes.append(mean_absolute_error(y_test, y_pred))
    results.append({
        "n_train": len(X_train) * prop,
        "mae_mean": np.mean(maes),
        "mae_std":  np.std(maes)
    })

# Full training set
rf = RandomForestRegressor(**rf_params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results.append({
    "n_train": len(X_train),
    "mae_mean": mean_absolute_error(y_test, y_pred),
    "mae_std": 0
    })

# Save in a dataframe
df_res = pd.DataFrame(results)
df_res.to_csv("./RESULTS/mae_vs_ntrain.csv", index=False)
print("Results saved in 'mae_vs_ntrain.csv'")

# Plot MAE vs n_train
plt.figure(figsize=(8,5))
plt.errorbar(
    df_res["n_train"], 
    
    df_res["mae_mean"],
    yerr=df_res["mae_std"],
    marker="o", linestyle="-", capsize=5
)
plt.xlabel("Training points")
plt.ylabel("Average MAE")
plt.title("MAE evolution vs. size of the training set")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("./RESULTS1/mae_vs_ntrain.png", dpi=300)
plt.show()
