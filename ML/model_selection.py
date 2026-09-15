import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, KFold

# Data loading
df = pd.read_csv("./DATA/train_val_set.csv", sep=";")
y_full = df["difference"].values
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

# Update dataset
X_full = X_full.drop(columns=to_drop)

# Results collect
results = []

# KFold
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Linear Regression + RFE
n_feats_total = X_full.shape[1]
for k in range(1, n_feats_total + 1):
    lr = LinearRegression()
    rfe = RFE(estimator=lr, n_features_to_select=k)
    rfe.fit(X_full, y_full)
    selected = X_full.columns[rfe.support_].tolist()
    X_sel = X_full[selected]
    
    # CV for MAE, R2
    cv_res_full = cross_validate(
        lr,
        X_sel,
        y_full,
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        n_jobs=-1
    )
    mae_cv_full = -np.mean(cv_res_full["test_mae"])
    r2_cv_full  = np.mean(cv_res_full["test_r2"])
    
    results.append({
        "model":    f"Linear_RFE_{k}",
        "MAE":      mae_cv_full,
        "R2":       r2_cv_full,
        "features": ", ".join(selected)
    })

# Lasso Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
lasso_cv = LassoCV(cv=cv, random_state=42)
lasso_cv.fit(X_scaled, y_full)
best_alpha = lasso_cv.alpha_

# a) CV for MAE, R2
lasso = LassoCV(alphas=[best_alpha], cv=cv, random_state=42)
cv_res_full = cross_validate(
    lasso,
    X_scaled,
    y_full,
    cv=cv,
    scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    n_jobs=-1
)
mae_cv_full = -np.mean(cv_res_full["test_mae"])
r2_cv_full  = np.mean(cv_res_full["test_r2"])

# Coefficient list
coef_series = pd.Series(lasso_cv.coef_, index=X_full.columns)
coef_sorted = coef_series.abs().sort_values(ascending=False).index.tolist()
coef_list = [f"{feat}:{coef_series[feat]:.4f}" for feat in coef_sorted]

results.append({
    "model":    "Lasso",
    "MAE":      mae_cv_full,
    "R2":       r2_cv_full,
    "features": ", ".join(coef_list)
})

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
cv_res_full = cross_validate(
    rf,
    X_full,
    y_full,
    cv=cv,
    scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
    n_jobs=-1
)
mae_cv_full = -np.mean(cv_res_full["test_mae"])
r2_cv_full  = np.mean(cv_res_full["test_r2"])

rf.fit(X_full, y_full)
importances = pd.Series(rf.feature_importances_, index=X_full.columns)
import_sorted = importances.sort_values(ascending=False)
imp_list = [f"{feat}:{import_sorted[feat]:.4f}" for feat in import_sorted.index]

results.append({
    "model":    "RandomForest",
    "MAE":      mae_cv_full,
    "R2":       r2_cv_full,
    "features": ", ".join(imp_list)
})

# Results saving
results_df = pd.DataFrame(results)
results_df.to_csv("./RESULTS/model_selection.csv", index=False)

print("Results saved in 'model_selection.csv'")