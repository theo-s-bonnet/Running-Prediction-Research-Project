import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Data loading
df_train = pd.read_csv("./DATA/train_val_set.csv", sep=";")
df_test     = pd.read_csv("./DATA/test_set.csv",    sep=";")

# Subsets
    # total
df_train_tot = df_train
df_test_tot = df_test

    # < 1 year
df_train_1year = df_train[df_train['period_duration'] <= 365]
df_test_1year = df_test[df_test['period_duration'] <= 365] 

    # < 2 years
df_train_2years = df_train[df_train['period_duration'] <= 730]
df_test_2years = df_test[df_test['period_duration'] <= 730] 

    # 12 weeks < period < 24 weeks
df_train_12_to_24_weeks = df_train[(df_train['period_duration'] >= 84)  & (df_train['period_duration'] <= 168)]
df_test_12_to_24_weeks = df_test[(df_test['period_duration'] >= 84)  & (df_test['period_duration'] <= 168)] 

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

# ===== X y TRAIN =====
    # total
X_train_tot = df_train_tot.drop(columns=to_drop)
y_train_tot = df_train_tot["difference"].values

    # < 1 year
X_train_1year = df_train_1year.drop(columns=to_drop)
y_train_1year = df_train_1year["difference"].values

    # < 2 years
X_train_2years = df_train_2years.drop(columns=to_drop)
y_train_2years = df_train_2years["difference"].values

    # 12 weeks < period < 24 weeks
X_train_12_to_24_weeks = df_train_12_to_24_weeks.drop(columns=to_drop)
y_train_12_to_24_weeks = df_train_12_to_24_weeks["difference"].values

# ===== X y TEST =====
    # total
X_test_tot = df_test_tot.drop(columns=to_drop)
y_test_tot = df_test_tot["difference"].values

    # < 1 year
X_test_1year = df_test_1year.drop(columns=to_drop)
y_test_1year = df_test_1year["difference"].values

    # < 2 years
X_test_2years = df_test_2years.drop(columns=to_drop)
y_test_2years = df_test_2years["difference"].values

    # 12 weeks < period < 24 weeks
X_test_12_to_24_weeks = df_test_12_to_24_weeks.drop(columns=to_drop)
y_test_12_to_24_weeks = df_test_12_to_24_weeks["difference"].values

# Verify similarity of columns
assert list(X_train_tot.columns) == list(X_test_tot.columns), "Columns don't correspond !"
assert list(X_train_1year.columns) == list(X_test_1year.columns), "Columns don't correspond !"
assert list(X_train_2years.columns) == list(X_test_2years.columns), "Columns don't correspond !"
assert list(X_train_12_to_24_weeks.columns) == list(X_test_12_to_24_weeks.columns), "Columns don't correspond !"

# Model
rf_final = RandomForestRegressor(
    n_estimators=1000,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=8,
    random_state=42,
    n_jobs=-1
)

# Predictions
# total
rf_final.fit(X_train_tot, y_train_tot)
y_pred_tot = rf_final.predict(X_test_tot)

# 1 year
rf_final.fit(X_train_1year, y_train_1year)
y_pred_1year = rf_final.predict(X_test_1year)

# 2 years
rf_final.fit(X_train_2years, y_train_2years)
y_pred_2years = rf_final.predict(X_test_2years)

# 12-24 weeks
rf_final.fit(X_train_12_to_24_weeks, y_train_12_to_24_weeks)
y_pred_12_to_24_weeks = rf_final.predict(X_test_12_to_24_weeks)

# Calcul of metrics
results = []

# total
mae_tot  = mean_absolute_error(y_test_tot, y_pred_tot)
r2_tot   = r2_score(y_test_tot, y_pred_tot)
results.append({
    "DATASET": "Total",
    "MAE":      mae_tot,
    "R2":       r2_tot,
})

# 1 year
mae_1year  = mean_absolute_error(y_test_1year, y_pred_1year)
r2_1year   = r2_score(y_test_1year, y_pred_1year)
results.append({
    "DATASET": "1 year max",
    "MAE":      mae_1year,
    "R2":       r2_1year,
})

# 2 years
mae_2years  = mean_absolute_error(y_test_2years, y_pred_2years)
r2_2years   = r2_score(y_test_2years, y_pred_2years)
results.append({
    "DATASET": "2 years max",
    "MAE":      mae_2years,
    "R2":       r2_2years,
})

# 12-24 weeks
mae_12_to_24_weeks  = mean_absolute_error(y_test_12_to_24_weeks, y_pred_12_to_24_weeks)
r2_12_to_24_weeks   = r2_score(y_test_12_to_24_weeks, y_pred_12_to_24_weeks)
results.append({
    "DATASET": "12 to 24 weeks",
    "MAE":      mae_12_to_24_weeks,
    "R2":       r2_12_to_24_weeks,
})

# Results saving
results = pd.DataFrame(results)
results.to_csv("./RESULTS1/model_test.csv", index=False)
print("Results saved in 'final_evaluation.csv'")
