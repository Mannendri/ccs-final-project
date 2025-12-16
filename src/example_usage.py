"""
Example of how to use the aggregated data for analysis and modeling
"""

import pandas as pd
from pathlib import Path

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Load the aggregated data
df = pd.read_csv(PROJECT_ROOT / 'data' / 'aggregated_data.csv')

# Convert timestamp column if needed
df['first_completion_time'] = pd.to_datetime(df['first_completion_time'], errors='coerce')

print("=== Data Overview ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(10))

print("\n=== Example: Get all tasks for a specific participant ===")
participant = 'kingfisher'
participant_data = df[df['participant_id'] == participant]
print(f"\nTasks for {participant}:")
print(participant_data[['task', 'perceived_difficulty', 'perceived_concreteness', 
                        'completed', 'days_late', 'task_points']])

print("\n=== Example: Compare perceived vs actual task characteristics ===")
# Tasks where perceived difficulty is higher than actual difficulty
# (Note: actual difficulty is categorical, so this is a simplified example)
print("\nTasks completed early (negative days_late):")
early_completions = df[df['days_late'] < 0]
print(early_completions[['participant_id', 'task', 'days_late', 'perceived_difficulty', 
                          'perceived_concreteness', 'task_points']])

print("\n=== Example: Calculate completion rates by perceived characteristics ===")
# Bin perceived ratings into low/medium/high
df['perceived_difficulty_bin'] = pd.cut(df['perceived_difficulty'], 
                                        bins=[0, 3, 6, 10], 
                                        labels=['low', 'medium', 'high'])
df['perceived_concreteness_bin'] = pd.cut(df['perceived_concreteness'], 
                                          bins=[0, 3, 6, 10], 
                                          labels=['low', 'medium', 'high'])

completion_by_difficulty = df.groupby('perceived_difficulty_bin')['completed'].agg(['sum', 'count', 'mean'])
print("\nCompletion by perceived difficulty:")
print(completion_by_difficulty)

completion_by_concreteness = df.groupby('perceived_concreteness_bin')['completed'].agg(['sum', 'count', 'mean'])
print("\nCompletion by perceived concreteness:")
print(completion_by_concreteness)

print("\n=== Example: Prepare data for modeling ===")
# Create a clean dataset for modeling (remove rows with missing ratings)
modeling_df = df.dropna(subset=['perceived_concreteness', 'perceived_difficulty', 
                                 'perceived_duration', 'perceived_reward']).copy()

# Create binary completion indicator
modeling_df['y_completed'] = modeling_df['completed'].astype(int)

# Create feature matrix (X) and target (y)
feature_cols = ['perceived_concreteness', 'perceived_difficulty', 
                'perceived_duration', 'perceived_reward', 'task_points']
X = modeling_df[feature_cols].values
y = modeling_df['y_completed'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Completion rate: {y.mean():.2%}")

# Example: Calculate correlation between perceived characteristics and completion
print("\n=== Correlations with completion ===")
for col in feature_cols:
    corr = modeling_df[col].corr(modeling_df['y_completed'])
    print(f"{col}: {corr:.3f}")

