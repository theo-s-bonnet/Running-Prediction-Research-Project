import os
import glob
import pandas as pd
import numpy as np
from itertools import groupby
from operator import itemgetter

# ------------ Config ------------
ACTIVITIES_SUMMARY_PATH = 'DATA/MEN/TEST/activities.csv'
COMPETITIONS_PATH       = 'DATA/MEN/TEST/competitions_filtered.csv'
DETAILED_DIR            = 'DATA/MEN/TEST/activities_corrected'
OUTPUT_PATH             = 'DATA/MEN/TEST/periods.csv'

# Thresholds
MAX_POINT_SPEED_KMH = 25             # km/h
MAX_POINT_SPEED_MS  = MAX_POINT_SPEED_KMH / 3.6  # m/s
MAX_DURATION_S      = 9.0            # s

# ------------ Utility Functions ------------
def load_competitions(path):
    """
    Load competitions from the competition file.
    Return a dataframe.
    """
    df = pd.read_csv(path, sep=';')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y_%m_%d')
    return df.sort_values('Date').reset_index(drop=True)

def load_activities_summary(path):
    """
    Load activity summary.
    Return a dataframe.
    """
    df = pd.read_csv(path)
    df['session_start_time'] = pd.to_datetime(
        df['session_start_time'], format='%Y_%m_%d %H:%M:%S'
    )
    return df[df['sport'].str.lower().str.startswith('run')].copy()

def load_detailed_activity(fp):
    """
    Load an activity file.
    Return a dataframe.
    """
    df = pd.read_csv(fp)
    return df

def points_to_speed_10k(points):
    """
    Approximate 10k speed from IAAF points.
    """
    # Approximate linear conversion
    return points * 0.00214233 + 3.511772926

def classify_zone(speed, v10k):
    """
    Return which intensity zone a speed belongs to.
    """
    pct = speed / v10k
    if pct < 0.82:
        return 1
    elif pct < 0.95:
        return 2
    else:
        return 3

def load_factor(speed, v10k):
    """
    Return the intensity factor for training load.
    """
    vma = v10k / 0.9
    delta = speed / vma
    return np.exp(5.3814 * (delta - 0.58))

# ------------ Core Analysis ------------
def analyze_period(period_id, start_idx, end_idx, comps_df, acts_df, detailed_dir):
    """
    Build period features.
    Return a dict containing values for one period.
    """
    # Define period
    start_date = comps_df.at[start_idx, 'Date']
    end_date   = comps_df.at[end_idx,   'Date']
    start_pts  = comps_df.at[start_idx, 'Points']
    end_pts    = comps_df.at[end_idx,   'Points']

    # Filter running activities
    mask = (
        (acts_df['session_start_time'] > start_date) &
        (acts_df['session_start_time'] <= end_date)
    )
    period_acts = acts_df.loc[mask].copy()
    if period_acts.empty:
        return None

    # Statistics
    period_duration    = (end_date - start_date).days + 1
    total_time         = period_acts['total_time'].sum()
    active_days        = period_acts['session_start_time'].dt.date.nunique()

    iso = period_acts['session_start_time'].dt.isocalendar()
    period_acts['week_id'] = (
        iso['year'].astype(str) + '-' + iso['week'].astype(str).str.zfill(2)
    )

    # Weekly time distribution
    weekly_time_ser   = period_acts.groupby('week_id')['total_time'].sum()
    weekly_time_dict  = weekly_time_ser.to_dict()
    avg_time_per_week = weekly_time_ser.mean()
    std_time_per_week = weekly_time_ser.std()
    active_weeks      = len(weekly_time_ser)

    # Days per week stats
    days_per_week       = period_acts.groupby('week_id')['session_start_time'] \
                                     .apply(lambda x: x.dt.date.nunique())
    mean_days_per_week  = days_per_week.mean()
    std_days_per_week   = days_per_week.std()

    # Consecutive active weeks
    week_starts = period_acts['session_start_time'] \
        .dt.to_period('W') \
        .dt.start_time \
        .drop_duplicates() \
        .sort_values()
    week_idxs = ((week_starts - week_starts.min()).dt.days // 7).tolist()
    sequences = [
        list(map(itemgetter(1), g))
        for k, g in groupby(enumerate(week_idxs), lambda x: x[0] - x[1])
    ]
    max_consec_weeks  = max(len(s) for s in sequences)
    nb_sequences      = len(sequences)

    # Zones and training load
    v10k = points_to_speed_10k(end_pts)
    zone_times = {1:0.0, 2:0.0, 3:0.0}
    total_load = 0.0

    # Mapping date->week_id
    date_to_week = {
        d: w for d, w in zip(
            period_acts['session_start_time'].dt.normalize(),
            period_acts['week_id']
        )
    }
    # Weekly load
    weekly_load_dict = {}

    for date in sorted(period_acts['session_start_time'].dt.normalize().unique()):
        week_id = date_to_week[date]
        pattern = os.path.join(detailed_dir, f"{date.strftime('%Y_%m_%d')}*.csv")
        for fp in sorted(glob.glob(pattern)):
            df = load_detailed_activity(fp)

            # Running filter
            if df.empty or not df['sport'].str.lower().iloc[0].startswith('run') :
                continue

            # Parsing timestamp & speed
            df['timestamp']      = pd.to_datetime(
                df['timestamp'], format='%Y_%m_%d %H:%M:%S', errors='coerce'
            )
            df = df.dropna(subset=['timestamp','enhanced_speed'])
            df['enhanced_speed'] = pd.to_numeric(df['enhanced_speed'], errors='coerce')
            df = df[df['enhanced_speed'] <= MAX_POINT_SPEED_MS]
            if len(df) < 2:
                continue

            # Duration between points
            df = df.sort_values('timestamp')
            df['duration'] = (
                df['timestamp'].diff().dt.total_seconds()
                .apply(lambda x: x if 0 < x <= MAX_DURATION_S else np.nan)
            ).fillna(0)

            # Zone & load
            df['zone'] = df['enhanced_speed'].apply(lambda s: classify_zone(s, v10k))
            df['load'] = df.apply(
                lambda r: load_factor(r['enhanced_speed'], v10k) * r['duration']/60,
                axis=1
            )

            # Agregate
            for z in (1,2,3):
                zone_times[z] += df.loc[df['zone']==z, 'duration'].sum()
            load_sum = df['load'].sum()
            total_load += load_sum
            weekly_load_dict[week_id] = weekly_load_dict.get(week_id, 0.0) + load_sum

    # 9) Return dict
    return {
        'Period_ID'           : f'P{period_id}',
        'start_perf'          : int(start_pts),
        'end_perf'            : int(end_pts),
        'difference'          : int(end_pts - start_pts),
        'period_duration'     : int(period_duration),
        'total_time'          : int(total_time),
        'avg_time_per_week'   : int(avg_time_per_week),
        'std_time_per_week'   : int(std_time_per_week),
        'active_days'         : int(active_days),
        'active_weeks'        : int(active_weeks),
        'mean_days_per_week'  : float(round(mean_days_per_week,2)),
        'std_days_per_week'   : float(round(std_days_per_week,2)),
        'max_consec_weeks'    : int(max_consec_weeks),
        'nb_sequences'        : int(nb_sequences),
        'zone1_time'          : int(zone_times[1]),
        'zone2_time'          : int(zone_times[2]),
        'zone3_time'          : int(zone_times[3]),
        'training_load'       : int(total_load),
        'avg_load_per_week'   : int((total_load) / (period_duration/7)) if period_duration>0 else 0,
        'std_load_per_week'   : int(np.std(list(weekly_load_dict.values()))),
        'weekly_time_dict'    : weekly_time_dict,
        'weekly_load_dict'    : weekly_load_dict
    }

# ------------ Generate CSV ------------
def generate_periods_csv():
    """
    Generate a .csv file containing all periods.
    """
    comps_df = load_competitions(path=COMPETITIONS_PATH)
    acts_df   = load_activities_summary(path=ACTIVITIES_SUMMARY_PATH)

    # 1) Consecutive periods
    simple = []
    pid = 1
    for i in range(len(comps_df)-1):
        print(str(i)+"/"+str(len(comps_df)-2))
        stats = analyze_period(pid, i, i+1, comps_df, acts_df, detailed_dir=DETAILED_DIR)
        if stats:
            simple.append(stats)
            pid += 1

    # 2) Combined periods
    records = list(simple)
    m = len(simple)
    for length in range(2, m+1):
        print("COMBO : "+str(length)+"/"+str(m))
        for start in range(0, m-length+1):
            chunk = simple[start:start+length]
            combo = {
                'Period_ID'         : '+'.join(p['Period_ID'] for p in chunk),
                'start_perf'        : chunk[0]['start_perf'],
                'end_perf'          : chunk[-1]['end_perf'],
                'difference'        : chunk[-1]['end_perf'] - chunk[0]['start_perf'],
                'period_duration'   : int(sum(p['period_duration'] for p in chunk)),
                'total_time'        : int(sum(p['total_time'] for p in chunk)),
                'active_days'       : int(sum(p['active_days'] for p in chunk)),
                'active_weeks'      : int(sum(p['active_weeks'] for p in chunk)),
                'zone1_time'        : int(sum(p['zone1_time'] for p in chunk)),
                'zone2_time'        : int(sum(p['zone2_time'] for p in chunk)),
                'zone3_time'        : int(sum(p['zone3_time'] for p in chunk)),
                'training_load'     : int(sum(p['training_load'] for p in chunk)),
                'avg_time_per_week' : int(sum(p['total_time'] for p in chunk) /
                                      (sum(p['period_duration'] for p in chunk)/7)),
                'avg_load_per_week' : int(sum(p['training_load'] for p in chunk) /
                                      (sum(p['period_duration'] for p in chunk)/7)),
                'mean_days_per_week': float(round(np.mean([p['mean_days_per_week'] for p in chunk]),2)),
                'std_days_per_week' : float(round(np.mean([p['std_days_per_week']  for p in chunk]),2)),
                'max_consec_weeks'  : int(max(p['max_consec_weeks'] for p in chunk)),
                'nb_sequences'      : int(sum(p['nb_sequences'] for p in chunk)),
            }
            
            agg_times = {}
            agg_loads = {}
            for p in chunk:
                for wk, t in p['weekly_time_dict'].items():
                    agg_times[wk] = agg_times.get(wk, 0) + t
                for wk, l in p['weekly_load_dict'].items():
                    agg_loads[wk] = agg_loads.get(wk, 0) + l
            combo['std_time_per_week'] = int(np.std(list(agg_times.values())))
            combo['std_load_per_week'] = int(np.std(list(agg_loads.values())))

            records.append(combo)

    # 3) Write of final CSV
    out_df = pd.DataFrame(records).drop(
        columns=['weekly_time_dict', 'weekly_load_dict']
    )
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generate {len(out_df)} periods in '{OUTPUT_PATH}'")

#=== RUN ===#
generate_periods_csv()
