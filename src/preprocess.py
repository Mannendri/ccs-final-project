"""
Preprocess and aggregate data from pre_study.csv and self_report.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Cohort start dates
COHORT1_START = datetime(2025, 12, 1)  # Monday Dec 1
COHORT2_START = datetime(2025, 12, 4)  # Thursday Dec 4

# Mapping from day name to relative day from start (for Cohort 1)
# Tuesday = Day 1, Thursday = Day 3, Saturday = Day 5
DEADLINE_RELATIVE_DAYS = {
    'Tuesday': 1,
    'Thursday': 3,
    'Saturday': 5
}

# Task information from proposal
# Using relative deadline days (from cohort start day = 0)
TASK_INFO = {
    'Sort a deck of cards': {
        'deadline': 'Tuesday',
        'deadline_day': 1,  # Day 1 from start
        'points': 5,
        'concreteness': 'high',
        'difficulty': 'easy',
        'duration': 'short'
    },
    'Complete a short quiz': {
        'deadline': 'Thursday',
        'deadline_day': 3,  # Day 3 from start
        'points': 5,
        'concreteness': 'high',
        'difficulty': 'easy',
        'duration': 'short'
    },
    'Complete fraternity chore': {
        'deadline': 'Saturday',
        'deadline_day': 5,  # Day 5 from start
        'points': 25,
        'concreteness': 'high',
        'difficulty': 'medium',
        'duration': 'long'
    },
    'Write a brief reflection': {
        'deadline': 'Tuesday',
        'deadline_day': 1,
        'points': 15,
        'concreteness': 'medium',
        'difficulty': 'medium',
        'duration': 'medium'
    },
    'Draw for ten minutes': {
        'deadline': 'Thursday',
        'deadline_day': 3,
        'points': 10,
        'concreteness': 'medium',
        'difficulty': 'easy',
        'duration': 'medium'
    },
    'Solve a logic puzzle': {
        'deadline': 'Saturday',
        'deadline_day': 5,
        'points': 15,
        'concreteness': 'medium',
        'difficulty': 'hard',
        'duration': 'medium'
    },
    'Explain a difficult concept from class': {
        'deadline': 'Tuesday',
        'deadline_day': 1,
        'points': 20,
        'concreteness': 'low',
        'difficulty': 'hard',
        'duration': 'medium'
    },
    'Analyze a short passage': {
        'deadline': 'Thursday',
        'deadline_day': 3,
        'points': 20,
        'concreteness': 'low',
        'difficulty': 'hard',
        'duration': 'long'
    }
}

# Task order in pre_study.csv (based on column headers)
TASK_ORDER = [
    'Sort a deck of cards',
    'Complete a short quiz',
    'Complete fraternity chore',
    'Write a brief reflection',
    'Draw for ten minutes',
    'Solve a logic puzzle',
    'Explain a difficult concept from class',
    'Analyze a short passage'
]


def normalize_participant_id(pid: str, preserve_suffixes: bool = True) -> str:
    """Normalize participant IDs to handle case differences.
    
    Args:
        pid: Participant ID to normalize
        preserve_suffixes: If True, preserve -1, -2 suffixes. If False, remove them.
    """
    # Convert to lowercase and strip whitespace
    pid = str(pid).lower().strip()
    
    # Preserve -1 and -2 suffixes (these indicate different participants)
    if preserve_suffixes:
        return pid
    else:
        # Remove suffixes for backward compatibility
        if '-' in pid:
            base_id = pid.split('-')[0]
            return base_id
        return pid


def determine_cohort(timestamp: pd.Timestamp) -> Tuple[datetime, int]:
    """Determine which cohort a participant belongs to based on timestamp.
    Returns (cohort_start_date, cohort_number)."""
    if timestamp < COHORT2_START:
        return COHORT1_START, 1
    else:
        return COHORT2_START, 2


def load_pre_study_data(filepath: str) -> pd.DataFrame:
    """Load and parse pre-study ratings data."""
    df = pd.read_csv(filepath)
    
    # Extract participant info
    participants = []
    
    # Skip header row (row 0) and process data rows
    for idx, row in df.iterrows():
        participant_id = normalize_participant_id(row['Anonymous ID'])
        timestamp = pd.to_datetime(row['Timestamp'])
        
        # Determine cohort
        cohort_start, cohort_num = determine_cohort(timestamp)
        start_day = 0  # Always 0 for the cohort start day
        
        # Extract ratings (32 total: 8 tasks × 4 dimensions)
        # Columns 2-9: Concreteness (8 tasks)
        # Columns 10-17: Difficulty (8 tasks)
        # Columns 18-25: Duration (8 tasks)
        # Columns 26-33: Reward (8 tasks)
        
        ratings = {}
        for i, task in enumerate(TASK_ORDER):
            ratings[task] = {
                'concreteness': float(row.iloc[2 + i]),
                'difficulty': float(row.iloc[10 + i]),
                'duration': float(row.iloc[18 + i]),
                'reward': float(row.iloc[26 + i])
            }
        
        participants.append({
            'participant_id': participant_id,
            'timestamp': timestamp,
            'cohort_start': cohort_start,
            'cohort_num': cohort_num,
            'start_day': start_day,
            'ratings': ratings
        })
    
    return pd.DataFrame(participants)


def load_self_report_data(filepath: str, participant_cohorts: Dict[str, datetime] = None) -> pd.DataFrame:
    """Load and parse self-report completion data.
    
    Args:
        filepath: Path to self_report.csv
        participant_cohorts: Optional dict mapping participant_id to cohort_start date.
                           If None, will infer from timestamp.
    """
    df = pd.read_csv(filepath)
    
    # Normalize participant IDs (preserve -1, -2 suffixes)
    df['participant_id'] = df['Anonymous Participant ID'].apply(
        lambda x: normalize_participant_id(x, preserve_suffixes=True)
    )
    df['timestamp'] = pd.to_datetime(df['Timestamp'])
    df['task'] = df['Which task are you logging?'].str.strip()
    df['completed'] = df['Did you fully complete the task?'].str.strip().str.lower() == 'yes'
    
    # Determine cohort for each row (infer from timestamp if not provided)
    def get_cohort_start(row):
        if participant_cohorts and row['participant_id'] in participant_cohorts:
            return participant_cohorts[row['participant_id']]
        else:
            # Infer from timestamp
            cohort_start, _ = determine_cohort(row['timestamp'])
            return cohort_start
    
    df['cohort_start'] = df.apply(get_cohort_start, axis=1)
    
    # Calculate relative day from cohort start (start day = 0)
    df['relative_day'] = (df['timestamp'] - df['cohort_start']).dt.days
    
    # Calculate days late using relative days (negative if completed early)
    def calculate_days_late(row):
        task_name = row['task']
        task_info = TASK_INFO.get(task_name)
        if task_info:
            deadline_day = task_info['deadline_day']  # Relative day from cohort start
            completion_day = row['relative_day']  # Relative day from cohort start
            days_late = completion_day - deadline_day
            return days_late
        return None
    
    df['days_late'] = df.apply(calculate_days_late, axis=1)
    
    # Calculate discounted points (10% per day late)
    def calculate_discounted_points(row):
        task_name = row['task']
        task_info = TASK_INFO.get(task_name)
        if task_info and row['days_late'] is not None:
            base_points = task_info['points']
            discount = max(0, row['days_late']) * 0.10  # 10% per day late
            discounted = base_points * (1 - discount)
            return max(0, discounted)  # Can't go negative
        return None
    
    df['discounted_points'] = df.apply(calculate_discounted_points, axis=1)
    
    # Add task metadata
    def add_task_metadata(row):
        task_name = row['task']
        task_info = TASK_INFO.get(task_name)
        if task_info:
            return pd.Series({
                'task_deadline': task_info['deadline'],
                'task_deadline_day': task_info['deadline_day'],  # Relative day from start
                'task_points': task_info['points'],
                'task_concreteness': task_info['concreteness'],
                'task_difficulty': task_info['difficulty'],
                'task_duration': task_info['duration']
            })
        return pd.Series({
            'task_deadline': None,
            'task_deadline_day': None,
            'task_points': None,
            'task_concreteness': None,
            'task_difficulty': None,
            'task_duration': None
        })
    
    metadata = df.apply(add_task_metadata, axis=1)
    df = pd.concat([df, metadata], axis=1)
    
    return df[['participant_id', 'timestamp', 'relative_day', 'task', 'completed', 'days_late', 
               'discounted_points', 'task_deadline', 'task_deadline_day', 'task_points', 
               'task_concreteness', 'task_difficulty', 'task_duration']]


def aggregate_data(pre_study_df: pd.DataFrame, self_report_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre-study ratings with completion data."""
    
    # Create a list to store aggregated records
    aggregated = []
    
    # Get unique participants from both datasets
    all_participants = set(pre_study_df['participant_id'].unique()) | set(self_report_df['participant_id'].unique())
    
    for participant_id in all_participants:
        # Get pre-study ratings for this participant
        pre_study_row = pre_study_df[pre_study_df['participant_id'] == participant_id]
        
        if len(pre_study_row) > 0:
            ratings = pre_study_row.iloc[0]['ratings']
        else:
            ratings = {}
        
        # Get all completions for this participant
        completions = self_report_df[self_report_df['participant_id'] == participant_id]
        
        # Create a record for each task
        for task_name in TASK_ORDER:
            # Get rating data
            task_ratings = ratings.get(task_name, {})
            
            # Get completion data (if any)
            task_completions = completions[completions['task'] == task_name]
            
            # Task metadata
            task_info = TASK_INFO.get(task_name, {})
            
            record = {
                'participant_id': participant_id,
                'task': task_name,
                'task_deadline': task_info.get('deadline'),
                'task_deadline_day': task_info.get('deadline_day'),  # Relative day from start
                'task_points': task_info.get('points'),
                'task_concreteness': task_info.get('concreteness'),
                'task_difficulty': task_info.get('difficulty'),
                'task_duration': task_info.get('duration'),
                # Perceived ratings
                'perceived_concreteness': task_ratings.get('concreteness'),
                'perceived_difficulty': task_ratings.get('difficulty'),
                'perceived_duration': task_ratings.get('duration'),
                'perceived_reward': task_ratings.get('reward'),
                # Completion data
                'completed': len(task_completions) > 0,
                'completion_count': len(task_completions),
                'first_completion_time': task_completions['timestamp'].min() if len(task_completions) > 0 else None,
                'relative_completion_day': task_completions['relative_day'].iloc[0] if len(task_completions) > 0 else None,
                'days_late': task_completions['days_late'].iloc[0] if len(task_completions) > 0 else None,
                'discounted_points': task_completions['discounted_points'].iloc[0] if len(task_completions) > 0 else None
            }
            
            aggregated.append(record)
    
    return pd.DataFrame(aggregated)


def main():
    """Main preprocessing function."""
    # Use paths relative to project root
    pre_study_path = PROJECT_ROOT / 'data' / 'pre_study.csv'
    self_report_path = PROJECT_ROOT / 'data' / 'self_report.csv'
    output_path = PROJECT_ROOT / 'data' / 'aggregated_data.csv'
    
    print("Loading pre-study data...")
    pre_study_df = load_pre_study_data(str(pre_study_path))
    print(f"Loaded {len(pre_study_df)} participant ratings")
    
    # Create mapping from participant_id to cohort_start for self-report data
    participant_cohorts = dict(zip(
        pre_study_df['participant_id'], 
        pre_study_df['cohort_start']
    ))
    
    print("\nLoading self-report data...")
    self_report_df = load_self_report_data(str(self_report_path), participant_cohorts)
    print(f"Loaded {len(self_report_df)} completion records")
    print(f"Unique participants: {self_report_df['participant_id'].unique()}")
    
    print("\nAggregating data...")
    aggregated_df = aggregate_data(pre_study_df, self_report_df)
    print(f"Created {len(aggregated_df)} aggregated records")
    
    # Save aggregated data
    aggregated_df.to_csv(output_path, index=False)
    print(f"\nSaved aggregated data to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nTotal participants: {aggregated_df['participant_id'].nunique()}")
    print(f"Total tasks per participant: {len(TASK_ORDER)}")
    print(f"\nCompletion rate: {aggregated_df['completed'].sum() / len(aggregated_df) * 100:.1f}%")
    print(f"\nTasks completed: {aggregated_df['completed'].sum()} / {len(aggregated_df)}")
    
    print("\n=== Completion by Task ===")
    task_completion = aggregated_df.groupby('task').agg({
        'completed': 'sum',
        'task_points': 'first'
    }).sort_values('task_points', ascending=False)
    print(task_completion)
    
    print("\n=== Average Perceived Ratings by Task ===")
    ratings_summary = aggregated_df.groupby('task').agg({
        'perceived_concreteness': 'mean',
        'perceived_difficulty': 'mean',
        'perceived_duration': 'mean',
        'perceived_reward': 'mean'
    }).round(2)
    print(ratings_summary)
    
    return aggregated_df, pre_study_df, self_report_df


if __name__ == '__main__':
    aggregated_df, pre_study_df, self_report_df = main()

