import os
import glob
import pandas as pd

def split_periods_by_athlete(root_dir="./DATA/MEN", 
                             train_val_file="./DATA/train_val_set.csv", 
                             test_file="./DATA/test_set.csv", 
                             test_frac=0.2, 
                             random_state=42):
    """
    For each file DATA/MEN/M*/periods.csv :
      - extract randomly `test_frac` from lines as test
      - the rest for train/validation
    Concatenate every piece to produce :
      - train_val_set.csv (80% of each athlete)
      - test_set.csv      (20% of each athlete)
    """
    pattern = os.path.join(root_dir, "M*", "periods.csv")
    period_files = sorted(glob.glob(pattern))

    if not period_files:
        print(f"No file 'periods.csv' found in {root_dir}/M*/")
        return

    train_val_dfs = []
    test_dfs      = []

    for fp in period_files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Impossible to read {fp} : {e}")
            continue

        if df.empty:
            continue

        # Random selection
        test_df = df.sample(frac=test_frac, random_state=random_state)
        # The rest in train/val
        train_val_df = df.drop(test_df.index)

        train_val_dfs.append(train_val_df)
        test_dfs.append(test_df)

    if not train_val_dfs or not test_dfs:
        print("No valid file")
        return

    # Concatenate all pieces
    train_val_global = pd.concat(train_val_dfs, ignore_index=True)
    test_global      = pd.concat(test_dfs,      ignore_index=True)

    # Save
    train_val_global.to_csv(train_val_file, index=False)
    test_global.to_csv(test_file, index=False)

    print(f"Created '{train_val_file}' with {len(train_val_global)} lines.")
    print(f"Created '{test_file}' with {len(test_global)} lines.")

#=== RUN ===#
split_periods_by_athlete()
