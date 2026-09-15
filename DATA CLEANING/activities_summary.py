import os
import csv
import glob

# Folder where activities files are
FOLDER_PATH = "./DATA/MEN/TEST/activities"

# Output filename
ACTIVITIES_FILENAME = "./DATA/MEN/TEST/activities.csv"

def generate_summary(folder_path, activities_filename):
    """
    Generate a .csv file summarizing all the activities
    """
    # Get all activities files (.csv)
    csv_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv"))]

    # List to store activities information
    activities = []

    for csv_file in csv_files:
        with open(csv_file, "r", newline="") as file:
            reader = csv.DictReader(file)
            first_row = next(reader, None)  # Read only first row
            
            if first_row:  # Verify that the file is not empty
                activities.append({
                    "session_start_time": first_row.get("session_start_time"),
                    "sport": first_row.get("sport"),
                    "total_time": first_row.get("total_time_session"),
                    "total_distance": first_row.get("total_distance_session"),
                    "total_ascent": first_row.get("total_ascent_session"),
                    "total_descent": first_row.get("total_descent_session"),
                })

    # Writing of the output file
    with open(activities_filename, "w", newline="") as file:
        fieldnames = ["session_start_time", "sport", "total_time", "total_distance", "total_ascent", "total_descent"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(activities)

    print(f"Summary file generated : {activities_filename}")

#=== RUN ===#
generate_summary(FOLDER_PATH, ACTIVITIES_FILENAME)
