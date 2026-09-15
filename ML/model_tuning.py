import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, r2_score

# Data loading
df = pd.read_csv("./DATA/train_val_set.csv", sep=";")
y = df["difference"].values
X_full = df.drop(columns=["difference", "Period_ID", "end_perf"])

# Delete very correlated features
# Correlation matrix (upper side)
corr_matrix = X_full.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Correlation with target
target_corr = df.drop(columns=["Period_ID", "end_perf"]).corr()["difference"].abs()

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

# Mise à jour du dataset
X = X_full.drop(columns=to_drop)

# KFolds
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Hyperparameters
param_grid = {
    "n_estimators":      [500, 1000, 1500],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2],
    "min_samples_leaf":  [1],
    "max_features":      [i for i in range(1,X.shape[1]+1)]
}
grid = list(ParameterGrid(param_grid))

N = len(grid)
i = 0

# Threshold for early stopping
threshold_neg_mae = -60 

results = []

# Loop on the grid
for params in grid:
    neg_mae_fold_sum = 0.0
    r2_fold_sum     = 0.0
    n_completed     = 0
    early_stop      = False

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        mae_fold = mean_absolute_error(y_test, y_pred)
        neg_mae_fold = -mae_fold
        r2_fold = r2_score(y_test, y_pred)

        neg_mae_fold_sum += neg_mae_fold
        r2_fold_sum     += r2_fold
        n_completed     += 1

        # Early stopping
        if fold_idx == 5:
            avg_neg_mae_first5 = neg_mae_fold_sum / 5
            if avg_neg_mae_first5 < threshold_neg_mae:
                early_stop = True
                break

    if early_stop:
        # Result saving
        results.append({
            **params,
            "mean_neg_mae": np.nan,
            "mean_r2":      np.nan,
            "abandoned":    True
        })
        i += 1
        print("STEP : "+str(i)+"/"+str(N))
        continue

    # If no early stopping, continue
    if n_completed < 10:
        for fold_idx2, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
            if fold_idx2 <= n_completed:
                continue
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            rf = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            neg_mae_fold_sum += -mean_absolute_error(y_test, y_pred)
            r2_fold_sum     += r2_score(y_test, y_pred)
        n_completed = 10

    mean_neg_mae = neg_mae_fold_sum / 10
    mean_r2      = r2_fold_sum / 10

    results.append({
        **params,
        "mean_neg_mae": mean_neg_mae,
        "mean_r2":      mean_r2,
        "abandoned":    False
    })

    i += 1
    print("STEP : "+str(i)+"/"+str(N))

# Dataframe
results_df = pd.DataFrame(results)

# Filter and convertion
results_df["MAE"] = -results_df["mean_neg_mae"]
results_df.loc[results_df["abandoned"], "MAE"] = np.nan

# Sorting
sorted_df = results_df.sort_values(by="MAE", na_position="last").reset_index(drop=True)

# Rank for MAE
sorted_df["rank_mae"] = sorted_df["MAE"].rank(method="dense")

# Select columns to save
param_cols = [c for c in sorted_df.columns if c.startswith("n_") or c.startswith("max_") or c.startswith("min_")]
output_cols = param_cols + ["MAE", "mean_r2", "abandoned", "rank_mae"]

sorted_df[output_cols].to_csv("./RESULTS/model_tuning.csv", index=False)

print("Tuning finished. Results saved in 'model_tuning.csv'.")