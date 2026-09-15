import os
import pandas as pd

# Folder where activities are
FOLDER_PATH = "DATA/MEN/TEST/activities"

def fill_missing_speed(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)

                # Verifie that columns exist
                if 'timestamp' not in df.columns or 'distance_record' not in df.columns:
                    continue

                # Convert timestamp in datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y_%m_%d %H:%M:%S', errors='coerce')
                df = df.dropna(subset=['timestamp'])

                # Verifie if "enhanced_speed" exist
                if 'enhanced_speed' not in df.columns:
                    df['enhanced_speed'] = 0.0

                # If enhanced_speed is unusable, it compute it
                if df['enhanced_speed'].sum() == 0:
                    # deltas time
                    df['delta_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

                    # deltas distance
                    df['delta_distance'] = df['distance_record'].diff().fillna(0)

                    # Calcul of speed
                    df['enhanced_speed'] = df.apply(
                        lambda row: row['delta_distance'] / row['delta_time'] if row['delta_time'] > 0 else 0,
                        axis=1
                    )

                    df.drop(columns=['delta_time', 'delta_distance'], inplace=True)

                    # Update
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y_%m_%d %H:%M:%S')
                    df.to_csv(file_path, index=False)
                    print(f"Updated: {filename}")
                else:
                    print(f"Already valid : {filename}")

            except Exception as e:
                print(f"Error on {filename} : {e}")

#=== RUN ===#
fill_missing_speed(FOLDER_PATH)
