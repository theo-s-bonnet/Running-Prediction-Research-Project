# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
# import numpy as np

# # === 1. Load dataset and target ===
# df = pd.read_csv("./DATA/train_val_set.csv", sep=";")
# y = df["difference"].values
# X = df.drop(columns=["difference", "Period_ID", "end_perf"])

# # === 2. Optionally remove highly correlated features (same logic as before) ===
# corr_matrix = X.corr().abs()
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# target_corr = df.drop(columns=["Period_ID", "end_perf"]).corr()["difference"].abs()

# to_drop = []
# for col in upper.columns:
#     for idx in upper.index:
#         if upper.loc[idx, col] > 0.90:
#             if target_corr[idx] < target_corr[col]:
#                 to_drop.append(idx)
#             else:
#                 to_drop.append(col)
# to_drop = list(set(to_drop))
# X = X.drop(columns=to_drop)

# # === 3. Fit Random Forest ===
# rf = RandomForestRegressor(n_estimators=500, random_state=42)
# rf.fit(X, y)

# # === 4. Get feature importances ===
# importances = pd.Series(rf.feature_importances_, index=X.columns)
# importances_sorted = importances.sort_values(ascending=False)

# # === 5. Plot ===
# plt.figure(figsize=(12, 8))
# sns.barplot(x=importances_sorted.values, y=importances_sorted.index, palette="viridis")
# plt.title("Feature Importances from Random Forest")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.show()

#==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# === 1. Load data ===
df = pd.read_csv("./DATA/train_val_set.csv", sep=";")
y = df["difference"].values
X = df.drop(columns=["difference", "Period_ID", "end_perf"])

# === 2. Optional: remove correlated features ===
# (Comment this out if already done previously)
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
target_corr = df.drop(columns=["Period_ID", "end_perf"]).corr()["difference"].abs()
to_drop = []
for col in upper.columns:
    for idx in upper.index:
        if upper.loc[idx, col] > 0.90:
            if target_corr[idx] < target_corr[col]:
                to_drop.append(idx)
            else:
                to_drop.append(col)
to_drop = list(set(to_drop))
X = X.drop(columns=to_drop)

# === 3. Split into train/test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 4. Train Random Forest ===
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# === 5. Predict on test set ===
y_pred = rf.predict(X_test)

# === 6. Plot predictions vs true values ===
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='y = x')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predictions vs True Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
