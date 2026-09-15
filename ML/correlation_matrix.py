import pandas as pd

# Data loading
df = pd.read_csv("./DATA/dataset.csv", sep=';')

# Correlation matrix
corr = df.corr(numeric_only=True)

# Save to a .csv
corr.to_csv("./RESULTS/correlation_matrix.csv")    

print("File generated")