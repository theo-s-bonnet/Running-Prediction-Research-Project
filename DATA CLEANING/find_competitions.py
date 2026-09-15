import csv
import re

INPUT_FILE = './DATA/MEN/TEST/strava.csv'
OUTPUT_FILE = './DATA/MEN/TEST/competitions.csv'

# Key-words
COMPETITION_KEYWORDS = [
    "championnat", "compétition", "marathon", "meeting", "semi", "finale",
    "10km", "5km", "1500m", "3000m", "5000m", "800m",
    "1 500", "3 000", "5 000", "1 500m", "3 000m", "5 000m"
]
COMP_KEYWORDS_LOWER = [kw.lower() for kw in COMPETITION_KEYWORDS]

def sanitize_text(text: str) -> str:
    """
    Correct description errors in .csv.
    """
    if text is None:
        return ""
    return re.sub(r'\s+', ' ', text.replace('\r', ' ').replace('\n', ' ')).strip()

def is_running_competition(activity: dict) -> bool:
    """
    Return True if an activity is a running activity and contains a key word.
    """
    # Get and normalize fields (FR / EN)
    type_act   = sanitize_text(activity.get("Type d'activité", activity.get("Activity Type", ""))).lower()
    name       = sanitize_text(activity.get("Nom de l'activité", activity.get("Activity Name", ""))).lower()
    desc       = sanitize_text(activity.get("Description de l'activité", activity.get("Activity Description", ""))).lower()

    if "course à pied" in type_act or "run" in type_act:
        return any(kw in name or kw in desc for kw in COMP_KEYWORDS_LOWER)
    return False

def extract_running_competitions(input_file: str, output_file: str):
    """
    Read strava.csv, filter running competitions and write fields in competitions.csv
    """
    selected_fields = [
        "Date de l'activité",
        "Nom de l'activité",
        "Description de l'activité",
        "Distance",
        "Temps écoulé"
    ]

    with open(input_file,  'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=selected_fields)
        writer.writeheader()

        for activity in reader:
            if is_running_competition(activity):
                clean_row = {
                    "Date de l'activité" : sanitize_text(activity.get("Date de l'activité", activity.get("Activity Date", ""))),
                    "Nom de l'activité" : sanitize_text(activity.get("Nom de l'activité", activity.get("Activity Name", ""))),
                    "Description de l'activité" : sanitize_text(activity.get("Description de l'activité", activity.get("Activity Description", ""))),
                    "Distance" : sanitize_text(activity.get("Distance", "")),
                    "Temps écoulé" :  sanitize_text(activity.get("Temps écoulé", activity.get("Elapsed Time", "")))
                }
                writer.writerow(clean_row)

    print(f"File {output_file} generated with running competitions")

#=== RUN ===#
extract_running_competitions(INPUT_FILE, OUTPUT_FILE)
