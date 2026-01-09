import numpy as np

# Example baseline efficiency table
# Replace later with CSV or API (nflfastR, collegefootballdata)
TEAM_EFFICIENCY = {
    "Bills": 0.62,
    "Chiefs": 0.64,
    "49ers": 0.66,
    "Eagles": 0.63,
    "Cowboys": 0.61,
}

def get_team_strength(team):
    return TEAM_EFFICIENCY.get(team, 0.50)

def matchup_probability(team_a, team_b, home=True):
    a = get_team_strength(team_a)
    b = get_team_strength(team_b)

    edge = a - b
    home_adj = 0.02 if home else 0

    prob = 0.5 + edge + home_adj
    return round(np.clip(prob, 0.40, 0.75), 3)
