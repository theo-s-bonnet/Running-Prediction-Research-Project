import os
import gzip
import shutil

# Folder where archives are
FOLDER_PATH = './DATA/MEN/TEST/activities'

def extract_files(directory):
    """
    Extract files from every .gz archives in the directory and delete archives
    """
    # Read all files and folder in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Verify if the file is a .gz archive
            if file.endswith('.gz'):
                gz_file_path = os.path.join(root, file)
                # Get filename
                file_path = gz_file_path[:-3]

                # Extract the .fit file from the .gz archive
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(file_path, 'wb') as fit_file:
                        shutil.copyfileobj(gz_file, fit_file)

                print(f"Extracted : {file_path}")

                # Delete .gz archive
                os.remove(gz_file_path)
                print(f"Deleted : {gz_file_path}")

#=== RUN ===#
extract_files(FOLDER_PATH)