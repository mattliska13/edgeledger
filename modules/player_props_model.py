import numpy as np

PLAYER_BASELINES = {
    "WR": {"targets": 7.2, "yards_per_target": 9.8},
    "RB": {"carries": 14.5, "yards_per_carry": 4.4},
    "QB": {"attempts": 33.0, "yards_per_attempt": 7.1},
}

def receiving_projection(position, snap_pct):
    base = PLAYER_BASELINES["WR"]
    targets = base["targets"] * snap_pct
    yards = targets * base["yards_per_target"]
    return round(targets, 1), round(yards, 1)

def rushing_projection(position, snap_pct):
    base = PLAYER_BASELINES["RB"]
    carries = base["carries"] * snap_pct
    yards = carries * base["yards_per_carry"]
    return round(carries, 1), round(yards, 1)

def passing_projection(snap_pct):
    base = PLAYER_BASELINES["QB"]
    attempts = base["attempts"] * snap_pct
    yards = attempts * base["yards_per_attempt"]
    return round(attempts, 1), round(yards, 1)
