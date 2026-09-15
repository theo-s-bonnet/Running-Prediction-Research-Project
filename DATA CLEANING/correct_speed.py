import os
import pandas as pd

# Parameters
speed_threshold = 26 / 3.6  # km/h to m/s


INPUT_DIR = "./DATA/MEN/TEST/activities"
OUTPUT_DIR = "./DATA/MEN/TEST/activities_corrected"

def correct_speeds(input_dir, output_dir):
    """
    Correct abnomalous speeds across all files in the input directory.
    Output a new directory with corrected files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        file_path = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(file_path)

            if 'enhanced_speed' not in df.columns or 'sport' not in df.columns:
                print(f"⚠️ Missing columns in {filename}")
                continue

            # Verify if the activity is a running activity
            sport_type = str(df['sport'].iloc[0]).lower()
            if sport_type != 'running':
                # Copy
                df.to_csv(os.path.join(output_dir, filename), index=False)
                continue

            df['enhanced_speed'] = pd.to_numeric(df['enhanced_speed'], errors='coerce')

            mask_aberrant = df['enhanced_speed'] > speed_threshold
            num_aberrant = mask_aberrant.sum()

            if num_aberrant == 0:
                df.to_csv(os.path.join(output_dir, filename), index=False)
                continue

            valid_speeds = df.loc[~mask_aberrant, 'enhanced_speed']
            if valid_speeds.empty:
                print(f"No valid speed in {filename}, file discarded")
                continue

            mean_speed = valid_speeds.mean()
            df.loc[mask_aberrant, 'enhanced_speed'] = mean_speed

            df.to_csv(os.path.join(output_dir, filename), index=False)
            print(f"Corrected : {filename} ({num_aberrant} speeds modified)")

        except Exception as e:
            print(f"Error with {filename} : {e}")
    

#=== RUN ===#
correct_speeds(INPUT_DIR, OUTPUT_DIR)