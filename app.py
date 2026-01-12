import os
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================
# Page + Style
# =========================
st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
}
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
.big-title { font-size: 1.9rem; font-weight: 900; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.subtle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.35rem; }
.card {
  background: #0b1220;
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.pill {
  display:inline-block; padding:0.18rem 0.55rem;
  border-radius:999px; background:rgba(255,255,255,0.08);
  margin-right:0.4rem; font-size:0.85rem;
}
.small {font-size:0.85rem; color:#94a3b8;}
hr { border: none; border-top: 1px solid rgba(148,163,184,0.18); margin: 10px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Keys / Secrets
# =========================
def get_key(name: str, default: str = "") -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        v = str(st.secrets.get(name, "")).strip()
        if v:
            return v
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

ODDS_API_KEY = get_key("ODDS_API_KEY", "d1a096c07dfb711c63560fcc7495fd0d")
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "")

# =========================
# HTTP session
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Dashboard/1.0 (streamlit)"})

def safe_get(url: str, params: dict, timeout: int = 25):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        ok = 200 <= r.status_code < 300
        try:
            payload = r.json()
        except Exception:
            payload = r.text
        return ok, r.status_code, payload, r.url
    except Exception as e:
        return False, 0, {"error": str(e)}, url

def is_list_of_dicts(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================
# Odds/EV helpers
# =========================
def american_to_implied(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def clamp01(x):
    return np.clip(x, 0.001, 0.999)

def pct_fmt(x01):
    return (pd.to_numeric(x01, errors="coerce") * 100.0).round(2)

def compute_ev_market_consensus(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    ModelProb = market consensus implied prob (avg across books for the same bet),
    EV = (ModelProb - implied(best price)) * 100.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Price"])

    for c in group_cols:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].astype("string")

    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))

    out["MarketProb"] = out.groupby(group_cols, dropna=False)["Implied"].transform("mean")
    out["MarketProb"] = clamp01(pd.to_numeric(out["MarketProb"], errors="coerce").fillna(out["Implied"]))

    out["ModelProb"] = out["MarketProb"]
    out["EV"] = (out["ModelProb"] - out["Implied"]) * 100.0
    return out

def best_price(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Price"])

    idx = out.groupby(group_cols, dropna=False)["Price"].idxmax()
    best = out.loc[idx].copy()
    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook", "Implied": "BestImplied"})
    best["EV"] = (pd.to_numeric(best["ModelProb"], errors="coerce") - pd.to_numeric(best["BestImplied"], errors="coerce")) * 100.0
    return best

def enforce_no_contradictions(df_best: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """
    Keep only the best EV row per key group.
    Example: totals -> don't show both Over and Under at same line.
    """
    if df_best.empty:
        return df_best

    out = df_best.copy()
    for c in key_cols:
        if c not in out.columns:
            out[c] = ""
        out[c] = out[c].astype("string")
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").fillna(-1e9)
    idx = out.groupby(key_cols, dropna=False)["EV"].idxmax()
    return out.loc[idx].sort_values("EV", ascending=False)

def plot_prob_bars(labels, probs01, title):
    fig = plt.figure()
    plt.barh(labels, probs01)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# =========================
# Odds API config (DK + FD only)
# =========================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"

SPORT_KEYS_GAME_LINES = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}
SPORT_KEYS_PROPS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

GAME_MARKETS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

# NOTE: These are the market tags you requested
PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_passing_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rushing_yds",
    "Receiving Yards": "player_receiving_yds",
    "Receptions": "player_receptions",
}

# =========================
# Sidebar
# =========================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Game Lines • Player Props • PGA</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA"], index=0)
debug = st.sidebar.checkbox("Show debug logs", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption("Best Price • EV ranking • NFL trends (if available) • No contradictory picks • Separate calls for Lines vs Props vs PGA")

# =============================================================================
# GAME LINES (separate call, cached)
# =============================================================================
@st.cache_data(ttl=60 * 60 * 24)  # one daily-ish cache
def fetch_game_lines(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": ",".join(GAME_MARKETS.values()),
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

def normalize_game_lines(raw):
    rows = []
    if not is_list_of_dicts(raw):
        return pd.DataFrame()

    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}"
        commence = ev.get("commence_time")

        for bm in ev.get("bookmakers", []) or []:
            book = bm.get("title") or bm.get("key")
            for mk in bm.get("markets", []) or []:
                mkey = mk.get("key")
                for out in mk.get("outcomes", []) or []:
                    rows.append({
                        "Event": matchup,
                        "Commence": commence,
                        "Market": mkey,
                        "Outcome": out.get("name"),
                        "Line": out.get("point"),
                        "Price": out.get("price"),
                        "Book": book,
                        "HomeTeam": home,
                        "AwayTeam": away,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["Market", "Outcome", "Price"])

# =============================================================================
# NFL Trends + Home/Away splits + Historical results (NFL only, optional)
# =============================================================================
@st.cache_data(ttl=60 * 60 * 24)
def load_nfl_results(season: int) -> pd.DataFrame:
    """
    Free NFL historical results via nfl_data_py (if installed).
    If not installed, return empty.
    """
    try:
        import nfl_data_py as nfl  # optional dependency
    except Exception:
        return pd.DataFrame()

    sch = nfl.import_schedules([season])
    keep = ["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]
    sch = sch[[c for c in keep if c in sch.columns]].copy()

    sch["home_score"] = pd.to_numeric(sch.get("home_score"), errors="coerce")
    sch["away_score"] = pd.to_numeric(sch.get("away_score"), errors="coerce")
    sch = sch.dropna(subset=["home_score", "away_score"], how="any")
    sch["gameday"] = pd.to_datetime(sch.get("gameday"), errors="coerce")
    sch = sch.dropna(subset=["gameday"])
    return sch

def compute_team_trends_nfl(results: pd.DataFrame, last_n: int = 5) -> pd.DataFrame:
    if results is None or results.empty:
        return pd.DataFrame()

    home = results[["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.rename(columns={"home_team": "team", "away_team": "opp", "home_score": "pf", "away_score": "pa"}, inplace=True)
    home["is_home"] = True

    away = results[["game_id", "gameday", "home_team", "away_team", "home_score", "away_score"]].copy()
    away.rename(columns={"away_team": "team", "home_team": "opp", "away_score": "pf", "home_score": "pa"}, inplace=True)
    away["is_home"] = False

    tg = pd.concat([home, away], ignore_index=True)
    tg["pd"] = tg["pf"] - tg["pa"]
    tg["win"] = (tg["pd"] > 0).astype(int)
    tg = tg.sort_values(["team", "gameday"])

    def agg_last(df):
        d = df.tail(last_n)
        dh = d[d["is_home"]]
        da = d[~d["is_home"]]
        return pd.Series({
            f"win_last{last_n}": d["win"].mean() if len(d) else np.nan,
            f"pd_last{last_n}": d["pd"].mean() if len(d) else np.nan,
            f"pf_last{last_n}": d["pf"].mean() if len(d) else np.nan,
            f"pa_last{last_n}": d["pa"].mean() if len(d) else np.nan,
            f"home_win_last{last_n}": dh["win"].mean() if len(dh) else np.nan,
            f"away_win_last{last_n}": da["win"].mean() if len(da) else np.nan,
            f"home_pd_last{last_n}": dh["pd"].mean() if len(dh) else np.nan,
            f"away_pd_last{last_n}": da["pd"].mean() if len(da) else np.nan,
        })

    trends = tg.groupby("team", as_index=False).apply(agg_last)
    if isinstance(trends.index, pd.MultiIndex):
        trends = trends.reset_index(drop=True)
    return trends

def attach_trends(df_lines: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    if df_lines.empty or trends is None or trends.empty:
        return df_lines

    out = df_lines.copy()
    home_cols = {c: f"Home_{c}" for c in trends.columns if c != "team"}
    away_cols = {c: f"Away_{c}" for c in trends.columns if c != "team"}

    out = out.merge(trends.rename(columns=home_cols).rename(columns={"team": "HomeTeam"}), on="HomeTeam", how="left")
    out = out.merge(trends.rename(columns=away_cols).rename(columns={"team": "AwayTeam"}), on="AwayTeam", how="left")
    return out

def apply_trend_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative trend adjustments to ModelProb for NFL:
    - Moneyline: recent win% and PD
    - Spreads: PD
    - Totals: expected total proxy from PF/PA
    """
    if df.empty:
        return df

    out = df.copy()
    for c in ["Home_win_last5", "Away_win_last5", "Home_pd_last5", "Away_pd_last5", "Home_pf_last5", "Home_pa_last5", "Away_pf_last5", "Away_pa_last5"]:
        if c not in out.columns:
            out[c] = np.nan

    def bump_ml(row):
        base = float(row["ModelProb"])
        hw = row.get("Home_win_last5", np.nan)
        aw = row.get("Away_win_last5", np.nan)
        hpd = row.get("Home_pd_last5", np.nan)
        apd = row.get("Away_pd_last5", np.nan)

        bump = 0.0
        if pd.notna(hw) and pd.notna(aw):
            bump += 0.07 * float(hw - aw)
        if pd.notna(hpd) and pd.notna(apd):
            bump += 0.02 * float((hpd - apd) / 7.0)

        outcome = str(row.get("Outcome", ""))
        home = str(row.get("HomeTeam", ""))
        if outcome == home:
            return float(np.clip(base + bump, 0.01, 0.99))
        return float(np.clip(base - bump, 0.01, 0.99))

    def bump_spread(row):
        base = float(row["ModelProb"])
        hpd = row.get("Home_pd_last5", np.nan)
        apd = row.get("Away_pd_last5", np.nan)
        if pd.isna(hpd) or pd.isna(apd):
            return base

        pd_diff = float(hpd - apd)
        bump = 0.02 * (pd_diff / 7.0)

        outcome = str(row.get("Outcome", ""))
        home = str(row.get("HomeTeam", ""))
        away = str(row.get("AwayTeam", ""))
        better = home if pd_diff > 0 else away
        return float(np.clip(base + bump, 0.01, 0.99)) if outcome == better else float(np.clip(base - bump, 0.01, 0.99))

    def bump_total(row):
        base = float(row["ModelProb"])
        line = row.get("Line", np.nan)
        if pd.isna(line):
            return base

        hpf = row.get("Home_pf_last5", np.nan)
        hpa = row.get("Home_pa_last5", np.nan)
        apf = row.get("Away_pf_last5", np.nan)
        apa = row.get("Away_pa_last5", np.nan)
        if any(pd.isna(x) for x in [hpf, hpa, apf, apa]):
            return base

        exp_home = 0.55 * float(hpf) + 0.45 * float(apa)
        exp_away = 0.55 * float(apf) + 0.45 * float(hpa)
        exp_total = exp_home + exp_away

        delta = (exp_total - float(line)) / 10.0
        bump = float(np.clip(0.06 * delta, -0.08, 0.08))

        outcome = str(row.get("Outcome", "")).lower()
        if outcome == "over":
            return float(np.clip(base + bump, 0.01, 0.99))
        if outcome == "under":
            return float(np.clip(base - bump, 0.01, 0.99))
        return base

    mk = out["Market"].astype(str).str.lower()
    ml_mask = mk.eq("h2h")
    sp_mask = mk.eq("spreads")
    to_mask = mk.eq("totals")

    out.loc[ml_mask, "ModelProb"] = out.loc[ml_mask].apply(bump_ml, axis=1)
    out.loc[sp_mask, "ModelProb"] = out.loc[sp_mask].apply(bump_spread, axis=1)
    out.loc[to_mask, "ModelProb"] = out.loc[to_mask].apply(bump_total, axis=1)

    out["EV"] = (out["ModelProb"] - out["Implied"]) * 100.0
    return out

# =============================================================================
# PLAYER PROPS (separate calls: events -> per event odds, robust 422 skipping)
# =============================================================================
@st.cache_data(ttl=60 * 60 * 24)  # one daily-ish cache
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 24)  # one daily-ish cache
def fetch_event_odds(sport_key: str, event_id: str, market_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

def normalize_player_props(event_payload: dict):
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    matchup = f"{away} @ {home}"

    for bm in event_payload.get("bookmakers", []) or []:
        book = bm.get("title") or bm.get("key")
        for mk in bm.get("markets", []) or []:
            mkey = mk.get("key")
            for out in mk.get("outcomes", []) or []:
                player = out.get("name")
                side = out.get("description")  # often Over/Under
                line = out.get("point")
                price = out.get("price")

                if player is None or price is None:
                    continue

                rows.append({
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": str(side) if side is not None else "",
                    "Line": line,
                    "Price": price,
                    "Book": book,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["Market", "Player", "Price"])

# =============================================================================
# DATAGOLF PGA LOGIC (course history / like courses proxy / recent form proxy / SG)
# =============================================================================
DG_BASE = "https://feeds.datagolf.com"

def dg_get(path: str, params: dict, timeout: int = 25):
    url = f"{DG_BASE}/{path.lstrip('/')}"
    ok, status, payload, final_url = safe_get(url, params=params, timeout=timeout)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

def dg_pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def fetch_dg_skill_and_preds():
    """
    Pulls:
    - Skill ratings (to extract SG T2G / SG Putting + optional recent form columns if present)
    - Pre-tournament predictions (often includes course-history/fit adjusted outputs)
    """
    # Skill ratings (schema can vary)
    skill = dg_get("preds/skill-ratings", params={"key": DATAGOLF_API_KEY, "file_format": "json"})

    df_skill = pd.DataFrame()
    if skill["ok"] and isinstance(skill["payload"], (dict, list)):
        payload = skill["payload"]
        if isinstance(payload, list):
            df_skill = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, list):
                    df_skill = pd.DataFrame(v)
                    break

    if not df_skill.empty:
        name_col = dg_pick_col(df_skill, ["player_name", "name", "player"])
        if name_col:
            df_skill = df_skill.rename(columns={name_col: "Player"})
        else:
            df_skill["Player"] = np.nan

    # Try to find SG columns
    sg_t2g_col = dg_pick_col(df_skill, ["sg_t2g", "sg_tee_to_green", "t2g", "strokes_gained_t2g"])
    sg_putt_col = dg_pick_col(df_skill, ["sg_putt", "sg_putting", "putt", "strokes_gained_putting"])

    # Optional recent-form style columns (varies; we'll detect if present)
    recent_form_col = dg_pick_col(df_skill, ["recent_form", "form", "skill_recent", "sg_total_last_50", "sg_total_last_24"])

    df_feat = pd.DataFrame()
    if not df_skill.empty:
        df_feat = df_skill[["Player"]].copy()
        if sg_t2g_col:
            df_feat["SG_T2G"] = pd.to_numeric(df_skill[sg_t2g_col], errors="coerce")
        if sg_putt_col:
            df_feat["SG_Putt"] = pd.to_numeric(df_skill[sg_putt_col], errors="coerce")
        if recent_form_col:
            df_feat["RecentForm"] = pd.to_numeric(df_skill[recent_form_col], errors="coerce")

    # Pre-tournament predictions (contains win/top10 probs; often includes course fit/history impacts)
    preds = dg_get(
        "preds/pre-tournament",
        params={
            "key": DATAGOLF_API_KEY,
            "tour": "pga",
            "odds_format": "percent",
            "file_format": "json",
            "add_position": "10",
            "dead_heat": "yes",
        },
    )

    df_preds = pd.DataFrame()
    if preds["ok"] and isinstance(preds["payload"], (dict, list)):
        payload = preds["payload"]
        if isinstance(payload, list):
            df_preds = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            for v in payload.values():
                if isinstance(v, list):
                    df_preds = pd.DataFrame(v)
                    break

    out_preds = pd.DataFrame()
    if not df_preds.empty:
        name_col = dg_pick_col(df_preds, ["player_name", "name", "player"])
        if name_col:
            df_preds = df_preds.rename(columns={name_col: "Player"})
        else:
            df_preds["Player"] = np.nan

        cols = set(df_preds.columns)

        def find_prob(keys):
            for k in keys:
                # exact
                for c in cols:
                    if c.lower() == k.lower():
                        return c
            # contains
            for k in keys:
                for c in cols:
                    if k.lower() in c.lower():
                        return c
            return None

        win_col = find_prob(["win", "win_prob", "prob_win"])
        top10_col = find_prob(["top_10", "top10", "top_10_prob", "top10_prob", "prob_top10"])

        # Try to find baseline vs course-fit/history versions (if DG provides)
        base_win_col = find_prob(["baseline_win", "base_win", "baseline_prob_win"])
        ch_win_col = find_prob(["course_win", "course_history_win", "course_fit_win", "baseline_plus_course_win"])

        base_top10_col = find_prob(["baseline_top10", "baseline_top_10", "base_top10"])
        ch_top10_col = find_prob(["course_top10", "course_history_top10", "course_fit_top10", "baseline_plus_course_top10"])

        def pct_to_prob(s):
            s = pd.to_numeric(s, errors="coerce")
            return s / 100.0

        out_preds = pd.DataFrame({"Player": df_preds["Player"]})
        if win_col:
            out_preds["DG_WinProb"] = pct_to_prob(df_preds[win_col])
        if top10_col:
            out_preds["DG_Top10Prob"] = pct_to_prob(df_preds[top10_col])

        # "Course history / like course fit" proxy: if DG exposes baseline vs course-adjusted, compute boost
        if base_win_col and ch_win_col:
            out_preds["CourseBoost_Win"] = pct_to_prob(df_preds[ch_win_col]) - pct_to_prob(df_preds[base_win_col])
        if base_top10_col and ch_top10_col:
            out_preds["CourseBoost_Top10"] = pct_to_prob(df_preds[ch_top10_col]) - pct_to_prob(df_preds[base_top10_col])

    return df_feat, out_preds, {"skill": skill, "preds": preds}

def normalize_dg_outrights(payload, market_label: str):
    """
    Defensive normalization for DG betting-tools/outrights.
    """
    if not isinstance(payload, (dict, list)):
        return pd.DataFrame()

    candidates = None
    if isinstance(payload, dict):
        for k in ["odds", "data", "players", "results"]:
            if k in payload and isinstance(payload[k], list):
                candidates = payload[k]
                break
        if candidates is None:
            for v in payload.values():
                if isinstance(v, list):
                    candidates = v
                    break
    else:
        candidates = payload

    if not isinstance(candidates, list):
        return pd.DataFrame()

    rows = []
    for p in candidates:
        if not isinstance(p, dict):
            continue

        name = p.get("player_name") or p.get("name") or p.get("player") or p.get("golfer")
        if not name:
            continue

        best_odds = p.get("best_odds") or p.get("best_price") or p.get("odds")
        best_book = p.get("best_book") or p.get("book") or p.get("sportsbook")

        # Some payloads include per-book arrays; compute best if needed
        if best_odds is None:
            for bk_key in ["books", "book_odds", "sportsbooks"]:
                if bk_key in p and isinstance(p[bk_key], list):
                    best = None
                    for b in p[bk_key]:
                        if not isinstance(b, dict):
                            continue
                        o = b.get("odds") or b.get("price")
                        if o is None:
                            continue
                        try:
                            o = float(o)
                        except Exception:
                            continue
                        score = o if o > 0 else -1e9 + o
                        if best is None or score > best["score"]:
                            best = {"score": score, "odds": o, "book": b.get("book") or b.get("sportsbook") or b.get("title")}
                    if best:
                        best_odds = best["odds"]
                        best_book = best["book"]
                    break

        rows.append({"Market": market_label, "Player": str(name), "BestBook": best_book, "BestOdds": best_odds})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["BestOdds"] = pd.to_numeric(df["BestOdds"], errors="coerce")
    return df.dropna(subset=["BestOdds", "Player"])

def pga_model_and_rank(allow_same_player_multi_markets: bool):
    win_rsp = dg_get(
        "betting-tools/outrights",
        params={"key": DATAGOLF_API_KEY, "tour": "pga", "market": "win", "odds_format": "american", "file_format": "json"},
    )
    top10_rsp = dg_get(
        "betting-tools/outrights",
        params={"key": DATAGOLF_API_KEY, "tour": "pga", "market": "top_10", "odds_format": "american", "file_format": "json"},
    )

    df_win = normalize_dg_outrights(win_rsp["payload"] if win_rsp["ok"] else {}, "Win")
    df_top10 = normalize_dg_outrights(top10_rsp["payload"] if top10_rsp["ok"] else {}, "Top-10")

    df = pd.concat([df_win, df_top10], ignore_index=True)
    if df.empty:
        return df, {"win": win_rsp, "top10": top10_rsp}

    df_feat, df_preds, dbg2 = fetch_dg_skill_and_preds()
    df = df.merge(df_feat, on="Player", how="left")
    df = df.merge(df_preds, on="Player", how="left")

    # Base probabilities from DG pre-tournament model if present
    df["BaseProb"] = np.nan
    df.loc[df["Market"] == "Win", "BaseProb"] = df.loc[df["Market"] == "Win", "DG_WinProb"]
    df.loc[df["Market"] == "Top-10", "BaseProb"] = df.loc[df["Market"] == "Top-10", "DG_Top10Prob"]

    # If missing, fall back to implied to avoid KeyErrors and nonsense
    df["Implied"] = df["BestOdds"].apply(american_to_implied)
    df["BaseProb"] = df["BaseProb"].fillna(df["Implied"])

    # Course history / like-course fit proxy boost (if DG provides baseline vs course-adjusted)
    course_boost = 0.0
    if "CourseBoost_Win" in df.columns:
        course_boost = course_boost + df["CourseBoost_Win"].fillna(0.0) * (df["Market"] == "Win")
    if "CourseBoost_Top10" in df.columns:
        course_boost = course_boost + df["CourseBoost_Top10"].fillna(0.0) * (df["Market"] == "Top-10")

    # Skill adjustment (small, conservative)
    # SG_T2G + SG_Putt + optional RecentForm
    def zscore(series):
        s = pd.to_numeric(series, errors="coerce")
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    z_t2g = zscore(df["SG_T2G"]) if "SG_T2G" in df.columns else 0.0
    z_putt = zscore(df["SG_Putt"]) if "SG_Putt" in df.columns else 0.0
    z_form = zscore(df["RecentForm"]) if "RecentForm" in df.columns else 0.0

    # Convert z to bounded adjustment
    adj = 0.010 * np.tanh(z_t2g) + 0.006 * np.tanh(z_putt) + 0.006 * np.tanh(z_form) + course_boost

    df["FinalProb"] = clamp01(pd.to_numeric(df["BaseProb"], errors="coerce").fillna(df["Implied"]) + adj).clip(0.0, 0.95)
    df["EV"] = (df["FinalProb"] - df["Implied"]) * 100.0

    # No contradictory picks:
    # If not allowed: keep only best market per player (either Win OR Top-10)
    if not allow_same_player_multi_markets:
        df = df.sort_values(["Player", "EV"], ascending=[True, False]).drop_duplicates("Player", keep="first")

    df = df.sort_values("EV", ascending=False)
    return df, {"win": win_rsp, "top10": top10_rsp, **dbg2}

# =============================================================================
# MODE: GAME LINES
# =============================================================================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sport = st.selectbox("Sport", list(SPORT_KEYS_GAME_LINES.keys()), index=0)
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1)
    top_n = st.slider("Top picks (EV)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    sport_key = SPORT_KEYS_GAME_LINES[sport]
    market_key = GAME_MARKETS[bet_type]

    # NFL-only trend controls
    use_trends = st.toggle("Use NFL trends + home/away splits + historical results (NFL only)", value=True)
    last_n = st.slider("Trend window (NFL)", 3, 10, 5)

    res = fetch_game_lines(sport_key)
    if debug:
        st.json({"endpoint": "odds(game_lines)", "status": res["status"], "url": res["url"], "params": res["params"]})

    if not res["ok"] or not is_list_of_dicts(res["payload"]):
        st.error("Game lines API did not return a valid list. Check ODDS_API_KEY / usage limits.")
        if debug:
            st.json(res["payload"])
        st.stop()

    df = normalize_game_lines(res["payload"])
    if df.empty:
        st.warning("No game line rows were normalized from the API response.")
        st.stop()

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        st.warning("No rows for this market right now.")
        st.stop()

    # EV baseline (market consensus)
    group_cols = ["Event", "Market", "Outcome"]
    if market_key in ["spreads", "totals"]:
        group_cols += ["Line"]

    df_ev = compute_ev_market_consensus(df, group_cols=group_cols)

    # Attach NFL trends and adjust model
    trends_note = ""
    if sport == "NFL" and use_trends:
        # try current + last season to ensure results exist in January
        year = datetime.now().year
        results = pd.concat([load_nfl_results(year - 1), load_nfl_results(year)], ignore_index=True).drop_duplicates()
        trends = compute_team_trends_nfl(results, last_n=last_n)
        if trends.empty:
            trends_note = "NFL trends unavailable (install nfl_data_py or no results loaded). Game Lines still work."
        else:
            df_ev = attach_trends(df_ev, trends)
            df_ev = apply_trend_adjustments(df_ev)
            trends_note = f"Applied NFL trends: last {last_n} games + home/away splits + historical results."

    elif use_trends and sport != "NFL":
        trends_note = "Trend logic is enabled for NFL only. Add a stats provider for CFB/CBB if you want splits/history there."

    df_best = best_price(df_ev, group_cols=group_cols)

    # No contradictions: one pick per event/market/line
    key_cols = ["Event", "Market"]
    if market_key in ["spreads", "totals"] and "Line" in df_best.columns:
        key_cols += ["Line"]
    df_best = enforce_no_contradictions(df_best, key_cols=key_cols)

    top_bets = df_best.sort_values("EV", ascending=False).head(int(top_n)).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["BestBook"].astype(str)

    disp = top_bets.copy()
    disp["ModelProb%"] = pct_fmt(disp["ModelProb"])
    disp["Implied%"] = pct_fmt(disp["BestImplied"])
    disp["EV"] = pd.to_numeric(disp["EV"], errors="coerce").round(2)

    st.subheader(f"{sport} — {bet_type}")
    if trends_note:
        st.caption(trends_note)

    cols = ["Event", "Outcome"] + (["Line"] if "Line" in disp.columns else []) + ["BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
    cols = [c for c in cols if c in disp.columns]
    st.dataframe(disp[cols], use_container_width=True, hide_index=True)

    st.markdown("#### EV (Top Picks)")
    chart_df = disp[["Event", "EV"]].copy().set_index("Event")
    st.bar_chart(chart_df)

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.sort_values("EV", ascending=False).head(25).copy()
        snap["⭐ Best Book"] = "⭐ " + snap["BestBook"].astype(str)
        snap["ModelProb%"] = pct_fmt(snap["ModelProb"])
        snap["Implied%"] = pct_fmt(snap["BestImplied"])
        snap["EV"] = pd.to_numeric(snap["EV"], errors="coerce").round(2)

        cols2 = ["Event", "Outcome"] + (["Line"] if "Line" in snap.columns else []) + ["BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MODE: PLAYER PROPS
# =============================================================================
elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0)
    prop_label = st.selectbox("Prop Type", list(PLAYER_PROP_MARKETS.keys()), index=0)

    top_n = st.slider("Top picks (EV)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    # Usage-control: scan only a few events daily
    max_events = st.sidebar.slider("Events to scan (usage control)", 1, 10, 5)

    sport_key = SPORT_KEYS_PROPS[sport]
    prop_market = PLAYER_PROP_MARKETS[prop_label]

    ev_res = fetch_events(sport_key)
    if debug:
        st.json({"endpoint": "events", "status": ev_res["status"], "url": ev_res["url"], "params": ev_res["params"]})

    if not ev_res["ok"] or not is_list_of_dicts(ev_res["payload"]):
        st.error("Events API did not return a valid list. Check ODDS_API_KEY / usage limits.")
        if debug:
            st.json(ev_res["payload"])
        st.stop()

    event_ids = [e.get("id") for e in ev_res["payload"] if isinstance(e, dict) and e.get("id")]
    event_ids = event_ids[: int(max_events)]
    if not event_ids:
        st.warning("No upcoming events found.")
        st.stop()

    all_rows = []
    call_log = []

    # Key: avoid 422 breaking the whole app -> skip event cleanly
    for eid in event_ids:
        r = fetch_event_odds(sport_key, eid, prop_market)
        call_log.append({"event_id": eid, "market": prop_market, "status": r["status"], "ok": r["ok"]})

        if not r["ok"] or not isinstance(r["payload"], dict):
            continue  # skip 422/401/etc

        dfp = normalize_player_props(r["payload"])
        if not dfp.empty:
            all_rows.append(dfp)

        time.sleep(0.08)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        st.warning("No props returned for DK/FD on scanned events (or this market isn’t available today). Try another prop type or scan more events.")
        st.stop()

    df = pd.concat(all_rows, ignore_index=True)

    # EV via market consensus
    group_cols = ["Event", "Market", "Player", "Side"]
    if "Line" in df.columns:
        group_cols += ["Line"]

    df_ev = compute_ev_market_consensus(df, group_cols=group_cols)
    df_best = best_price(df_ev, group_cols=group_cols)

    # No contradictions: one pick per (Event, Market, Player, Line) -> avoid Over AND Under for same player line
    key_cols = ["Event", "Market", "Player"]
    if "Line" in df_best.columns:
        key_cols += ["Line"]
    df_best = enforce_no_contradictions(df_best, key_cols=key_cols)

    df_best = df_best.sort_values("EV", ascending=False)
    top_bets = df_best.head(int(top_n)).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["BestBook"].astype(str)

    disp = top_bets.copy()
    disp["ModelProb%"] = pct_fmt(disp["ModelProb"])
    disp["Implied%"] = pct_fmt(disp["BestImplied"])
    disp["EV"] = pd.to_numeric(disp["EV"], errors="coerce").round(2)

    st.subheader(f"{sport} — Player Props ({prop_label})")
    st.caption("Best price across DK/FD • ModelProb = market consensus • EV uses best implied probability from odds")

    cols = ["Event", "Player", "Side"] + (["Line"] if "Line" in disp.columns else []) + ["BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
    cols = [c for c in cols if c in disp.columns]
    st.dataframe(disp[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Model vs Implied (Top Bets)")
    chart = disp[["Player", "ModelProb", "BestImplied"]].copy().set_index("Player")
    plot_prob_bars(chart.index.tolist(), chart["ModelProb"].values, "Model Probability (Top Bets)")
    plot_prob_bars(chart.index.tolist(), chart["BestImplied"].values, "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.head(25).copy()
        snap["⭐ Best Book"] = "⭐ " + snap["BestBook"].astype(str)
        snap["ModelProb%"] = pct_fmt(snap["ModelProb"])
        snap["Implied%"] = pct_fmt(snap["BestImplied"])
        snap["EV"] = pd.to_numeric(snap["EV"], errors="coerce").round(2)

        cols2 = ["Event", "Player", "Side"] + (["Line"] if "Line" in snap.columns else []) + ["BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# MODE: PGA (DataGolf)
# =============================================================================
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🏌️ PGA — Win + Top-10 (Course History • Like Courses • Recent Form • SG T2G • Putting • Odds/EV)")

    if not DATAGOLF_API_KEY:
        st.error('Missing DATAGOLF_API_KEY. Add it in Streamlit Secrets as DATAGOLF_API_KEY="..."')
        st.stop()

    allow_multi = st.toggle("Allow same golfer in multiple markets (Win + Top-10)", value=False)
    top_n = st.slider("Top picks (EV)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    with st.spinner("Loading DataGolf odds + probabilities…"):
        df_all, dbg = pga_model_and_rank(allow_same_player_multi_markets=allow_multi)

    if df_all.empty:
        st.warning("No PGA odds/probabilities returned right now.")
        if debug:
            st.json(dbg)
        st.stop()

    # Top N
    top = df_all.head(int(top_n)).copy()
    top["FinalProb%"] = pct_fmt(top["FinalProb"])
    top["Implied%"] = pct_fmt(top["Implied"])
    top["EV"] = pd.to_numeric(top["EV"], errors="coerce").round(2)

    st.markdown("### Top Picks")
    cols = ["Market", "Player", "BestOdds", "BestBook", "FinalProb%", "Implied%", "EV"]
    for extra in ["SG_T2G", "SG_Putt", "RecentForm", "CourseBoost_Win", "CourseBoost_Top10"]:
        if extra in top.columns:
            cols.append(extra)
    cols = [c for c in cols if c in top.columns]

    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probabilities (Top Picks)")
    plot_prob_bars(top["Player"].tolist(), (pd.to_numeric(top["FinalProb"], errors="coerce").fillna(0.0)).values, "Final Model Probability")
    plot_prob_bars(top["Player"].tolist(), (pd.to_numeric(top["Implied"], errors="coerce").fillna(0.0)).values, "Implied Probability (Odds)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_all.head(25).copy()
        snap["FinalProb%"] = pct_fmt(snap["FinalProb"])
        snap["Implied%"] = pct_fmt(snap["Implied"])
        snap["EV"] = pd.to_numeric(snap["EV"], errors="coerce").round(2)

        cols2 = ["Market", "Player", "BestOdds", "BestBook", "FinalProb%", "Implied%", "EV"]
        for extra in ["SG_T2G", "SG_Putt", "RecentForm", "CourseBoost_Win", "CourseBoost_Top10"]:
            if extra in snap.columns:
                cols2.append(extra)
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.caption(
        "Notes: Course history / like-course fit is applied as a conservative boost when DataGolf provides baseline vs course-adjusted outputs "
        "(exposed as CourseBoost_* columns when available). SG T2G / SG Putting and optional RecentForm (if present in skill ratings) are used as small adjustments."
    )

    if debug:
        with st.expander("Debug"):
            st.json({
                "win_status": {"ok": dbg.get("win", {}).get("ok"), "status": dbg.get("win", {}).get("status"), "url": dbg.get("win", {}).get("url")},
                "top10_status": {"ok": dbg.get("top10", {}).get("ok"), "status": dbg.get("top10", {}).get("status"), "url": dbg.get("top10", {}).get("url")},
                "skill_status": {"ok": dbg.get("skill", {}).get("ok"), "status": dbg.get("skill", {}).get("status"), "url": dbg.get("skill", {}).get("url")},
                "preds_status": {"ok": dbg.get("preds", {}).get("ok"), "status": dbg.get("preds", {}).get("status"), "url": dbg.get("preds", {}).get("url")},
            }, expanded=False)

    st.markdown("</div>", unsafe_allow_html=True)
