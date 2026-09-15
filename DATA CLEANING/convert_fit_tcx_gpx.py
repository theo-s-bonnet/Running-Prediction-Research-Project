import pandas as pd
import os
import fitdecode
import xml.etree.ElementTree as ET
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

# Folder where activities are
FOLDER_PATH = "./DATA/MEN/TEST/activities"

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance in meters between two latitude/longitude points
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def parse_fit(file_path):
    """
    Custom parser for .fit file
    """

    session_data = {
        "session_start_time": None,
        "sport": None,
        "total_time_session": None,
        "total_distance_session": None,
        "total_ascent_session": None,
        "total_descent_session": None
    }

    lap_data = {"total_timer_time_lap": None, "total_distance_lap": None, "total_ascent_lap": None}

    data_rows = []
    raw_datetime = None

    # Lap
    with fitdecode.FitReader(file_path) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue
            if frame.name == "session":
                fields = {f.name: f.value for f in frame.fields}
                raw_datetime = fields.get("start_time")
                if raw_datetime:
                    session_data["session_start_time"] = raw_datetime.strftime("%Y_%m_%d %H:%M:%S")
                session_data["sport"] = str(fields.get("sport")).capitalize()
                session_data["total_time_session"] = fields.get("total_timer_time")
                session_data["total_distance_session"] = fields.get("total_distance")
                session_data["total_ascent_session"] = fields.get("total_ascent")
                session_data["total_descent_session"] = fields.get("total_descent")
            elif frame.name == "lap":
                fields = {f.name: f.value for f in frame.fields}
                lap_data.update({
                    "total_timer_time_lap": fields.get("total_timer_time"),
                    "total_distance_lap": fields.get("total_distance"),
                    "total_ascent_lap": fields.get("total_ascent")
                })

    # Recording points 
    with fitdecode.FitReader(file_path) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                continue
            fields = {f.name: f.value for f in frame.fields}
            timestamp = fields.get("timestamp")
            if not timestamp:
                continue
            data_rows.append([
                timestamp.strftime("%Y_%m_%d %H:%M:%S"),
                session_data["session_start_time"],
                session_data["sport"],
                session_data["total_time_session"],
                session_data["total_distance_session"],
                session_data["total_ascent_session"],
                session_data["total_descent_session"],
                lap_data["total_timer_time_lap"],
                lap_data["total_distance_lap"],
                lap_data["total_ascent_lap"],
                fields.get("distance"),
                fields.get("enhanced_speed"),
                fields.get("enhanced_altitude")
            ])

    return raw_datetime, data_rows


def parse_tcx(file_path):
    """
    Custom parser for .tcx file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().lstrip()
    root = ET.fromstring(content)
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    activity = root.find(".//tcx:Activity", ns)
    if activity is None:
        raise ValueError("Aucune activité trouvée dans le fichier TCX.")

    session_id = activity.find("tcx:Id", ns).text
    try:
        session_start_time = datetime.strptime(session_id, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try : 
            session_start_time = datetime.strptime(session_id, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError :
            session_start_time = datetime.strptime(session_id, "%Y-%m-%dT%H:%M:%S%z")
    sport = activity.get("Sport")
    laps = activity.findall("tcx:Lap", ns)
    total_timer_session = total_distance_session = total_ascent_session = total_descent_session = 0
    data_rows = []
    for lap in laps:
        total_timer_lap = float(lap.find("tcx:TotalTimeSeconds", ns).text)
        try :
            total_distance_lap = float(lap.find("tcx:DistanceMeters", ns).text)
        except AttributeError :
            total_distance_lap = 0

        total_ascent_lap = 0
        total_descent_lap = 0

        total_timer_session += total_timer_lap
        total_distance_session += total_distance_lap
        total_ascent_session += total_ascent_lap
        total_descent_session += total_descent_lap

        records = lap.findall(".//tcx:Trackpoint", ns)
        for record in records[:-1]:
            timestamp = record.find("tcx:Time", ns).text
            try :
                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError :
                try : 
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError :
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
            distance_record = record.find("tcx:DistanceMeters", ns)
            enhanced_speed = record.find(".//tcx:Speed", ns)
            enhanced_altitude = record.find("tcx:AltitudeMeters", ns)

            distance_record = float(distance_record.text) if distance_record is not None else 0
            enhanced_speed = float(enhanced_speed.text) if enhanced_speed is not None else 0
            enhanced_altitude = float(enhanced_altitude.text) if enhanced_altitude is not None else 0

            data_rows.append([
                timestamp.strftime("%Y_%m_%d %H:%M:%S"), session_start_time.strftime("%Y_%m_%d %H:%M:%S"), sport,
                None, None, None, None,
                None, None,
                total_ascent_lap, distance_record, enhanced_speed, enhanced_altitude
            ])

        if len(records) != 0:
            record = records[-1]
            timestamp = record.find("tcx:Time", ns).text
            try :
                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                try :
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                except ValueError :
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
            distance_record = record.find("tcx:DistanceMeters", ns)
            enhanced_speed = record.find(".//tcx:Speed", ns)
            enhanced_altitude = record.find("tcx:AltitudeMeters", ns)

            distance_record = float(distance_record.text) if distance_record is not None else 0
            enhanced_speed = float(enhanced_speed.text) if enhanced_speed is not None else 0
            enhanced_altitude = float(enhanced_altitude.text) if enhanced_altitude is not None else 0

            data_rows.append([
                timestamp.strftime("%Y_%m_%d %H:%M:%S"), session_start_time.strftime("%Y_%m_%d %H:%M:%S"), sport,
                None, None, None, None,
                total_timer_lap, total_distance_lap,
                total_ascent_lap, distance_record, enhanced_speed, enhanced_altitude
            ])

    for row in data_rows:
        row[3] = total_timer_session
        row[4] = total_distance_session
        row[5] = total_ascent_session
        row[6] = total_descent_session

    return session_start_time, data_rows


def parse_gpx(file_path):
    """
    Custom parser for .gpx file
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    # generate namespace
    ns_uri = root.tag[root.tag.find('{')+1:root.tag.find('}')] if '}' in root.tag else ''
    ns = {'ns': ns_uri} if ns_uri else {}

    # session start
    time_elem = root.find('ns:metadata/ns:time', ns)
    if time_elem is None :
        time_elem = root.find('metadata/time')
    session_start = time_elem.text
    try:
        session_start_dt = datetime.strptime(session_start, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        try :
            session_start_dt = datetime.strptime(session_start, "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError :
            session_start_dt = datetime.strptime(session_start, "%Y-%m-%dT%H:%M:%S%z")
    session_str = session_start_dt.strftime("%Y_%m_%d %H:%M:%S")

    # sport
    sport_elem = root.find('ns:trk/ns:type', ns)
    if sport_elem is None :
        sport_elem = root.find('trk/type')
    sport = sport_elem.text.capitalize() if sport_elem is not None else 'Running'

    # collect points
    trkpts = root.findall('.//ns:trkpt', ns)
    if trkpts is None :
        trkpts = root.findall('.//trkpt')
    points = []
    for pt in trkpts:
        lat = float(pt.get('lat'))
        lon = float(pt.get('lon'))
        ele_elem = pt.find('ns:ele', ns)
        if ele_elem is None :
            ele_elem = pt.find('ele')
        ele = float(ele_elem.text) if ele_elem is not None else None
        time_elem = pt.find('ns:time', ns)
        if time_elem is None :
            time_elem = pt.find('time')
        t = time_elem.text
        try:
            dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            try :
                dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError :
                dt = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z")
        points.append({'time': dt, 'lat': lat, 'lon': lon, 'ele': ele})

    if not points:
        raise ValueError("Aucun point GPS trouvé dans le GPX.")

    points.sort(key=lambda x: x['time'])
    # calculs session
    total_dist = 0.0
    total_asc = 0.0
    total_desc = 0.0
    prev = points[0]
    for pt in points[1:]:
        d = haversine(prev['lat'], prev['lon'], pt['lat'], pt['lon'])
        total_dist += d
        if prev['ele'] is not None and pt['ele'] is not None:
            de = pt['ele'] - prev['ele']
            if de > 0: total_asc += de
            else: total_desc += abs(de)
        prev = pt
    total_time = (points[-1]['time'] - points[0]['time']).total_seconds()

    # construct data_rows
    data_rows = []
    cum_dist = 0.0
    prev = points[0]
    # first point
    data_rows.append([
        prev['time'].strftime("%Y_%m_%d %H:%M:%S"), session_str, sport,
        total_time, total_dist, total_asc, total_desc,
        None, None, None,
        cum_dist, 0, prev['ele']
    ])
    for pt in points[1:]:
        d = haversine(prev['lat'], prev['lon'], pt['lat'], pt['lon'])
        dt = (pt['time'] - prev['time']).total_seconds()
        speed = d / dt if dt > 0 else 0
        cum_dist += d
        data_rows.append([
            pt['time'].strftime("%Y_%m_%d %H:%M:%S"), session_str, sport,
            total_time, total_dist, total_asc, total_desc,
            None, None, None,
            cum_dist, speed, pt['ele']
        ])
        prev = pt

    return session_start_dt, data_rows


def generate_filename(session_start_time, folder_path):
    """
    Generate filename based on activity date
    """
    date_part = session_start_time.strftime("%Y_%m_%d")
    existing = [f for f in os.listdir(folder_path) if f.startswith(date_part)]
    return f"{date_part}_{len(existing)+1:02d}.csv"


def convert_file_to_csv(file_path, folder_path):
    """
    Convert data rows to .csv file
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.fit':
        session_dt, rows = parse_fit(file_path)
    elif ext == '.tcx':
        session_dt, rows = parse_tcx(file_path)
    elif ext == '.gpx':
        session_dt, rows = parse_gpx(file_path)
    else:
        raise ValueError("Format non supporté")

    df = pd.DataFrame(rows, columns=[
        "timestamp", "session_start_time", "sport",
        "total_time_session", "total_distance_session",
        "total_ascent_session", "total_descent_session",
        "total_timer_time_lap", "total_distance_lap",
        "total_ascent_lap", "distance_record",
        "enhanced_speed", "enhanced_altitude"
    ])

    out = os.path.join(folder_path, generate_filename(session_dt, folder_path))
    df.to_csv(out, index=False)
    print(f"Converted {file_path} → {out}")

    if ext in ('.fit', '.tcx', '.gpx'):
        os.remove(file_path)


def convert_folder(folder_path):
    """
    Global loop across files
    """
    for fname in os.listdir(folder_path):
        if os.path.splitext(fname)[1].lower() in ('.fit', '.tcx', '.gpx'):
            convert_file_to_csv(os.path.join(folder_path, fname), folder_path)

#=== RUN ===#
convert_folder(FOLDER_PATH)
