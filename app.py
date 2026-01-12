# app.py
import os
from datetime import datetime, timezone
import math
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Hide Streamlit default menu/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar polish */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
}
section[data-testid="stSidebar"] * {
  color: #e5e7eb !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
}
h1, h2, h3 {
  letter-spacing: -0.02em;
}
.big-title {
  font-size: 2.0rem;
  font-weight: 800;
  margin-bottom: 0.2rem;
}
.subtle {
  color: #94a3b8;
  font-size: 0.95rem;
}
.card {
  background: #0b1220;
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 14px 16px;
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.25);
  font-size: 0.8rem;
  margin-left: 8px;
  color: #e5e7eb;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Secrets / Keys
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    # Streamlit Cloud: st.secrets first
    if hasattr(st, "secrets") and name in st.secrets:
        v = str(st.secrets.get(name, "")).strip()
        if v:
            return v
    # env var fallback
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

ODDS_API_KEY = get_secret("ODDS_API_KEY", "")  # your new Odds API key (recommended in secrets)
DATAGOLF_API_KEY = get_secret("DATAGOLF_API_KEY", "")  # your DataGolf key

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.markdown("## **Dashboard**")
st.sidebar.markdown("<div class='subtle'>Game lines + PGA models</div>", unsafe_allow_html=True)

section = st.sidebar.radio("Section", ["Game Lines", "PGA (Win + Top-10)"], index=0)

debug = st.sidebar.toggle("Show debug", value=False)

# -----------------------------
# Helpers
# -----------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Dashboard/1.0 (+streamlit)"})

def safe_get(url: str, params: dict, timeout: int = 20):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        ok = (200 <= r.status_code < 300)
        payload = None
        err = None
        try:
            payload = r.json()
        except Exception:
            payload = r.text
        if not ok:
            err = {"status": r.status_code, "payload": payload}
        return ok, r.status_code, payload, err
    except Exception as e:
        return False, 0, None, {"exception": str(e)}

def american_to_implied(odds: float) -> float:
    # returns probability in [0,1]
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o == 0:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def implied_to_american(p: float) -> float:
    # p in (0,1). convert to american.
    p = max(1e-9, min(1 - 1e-9, float(p)))
    if p >= 0.5:
        return -round(100 * p / (1 - p), 0)
    return round(100 * (1 - p) / p, 0)

def no_vig_two_way(p1: float, p2: float):
    # normalize two probabilities so they sum to 1
    s = (p1 or 0) + (p2 or 0)
    if s <= 0:
        return np.nan, np.nan
    return p1 / s, p2 / s

def softmax(x: np.ndarray, temp: float = 1.0):
    x = np.array(x, dtype=float)
    x = x / max(1e-9, float(temp))
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s

def as_utc(dt_str: str):
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

# -----------------------------
# GAME LINES (NFL/CFB/CBB)
# -----------------------------
ODDS_SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}

BOOKMAKERS = "draftkings,fanduel"
GAME_MARKETS = ["h2h", "spreads", "totals"]

@st.cache_data(ttl=60 * 60 * 12)  # 12 hours cache
def fetch_game_lines(sport_key: str):
    if not ODDS_API_KEY:
        return {"ok": False, "error": "Missing ODDS_API_KEY", "events": []}

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, err = safe_get(url, params=params)
    if not ok:
        return {"ok": False, "status": status, "error": err, "events": []}
    if not isinstance(payload, list):
        return {"ok": False, "status": status, "error": {"payload_type": type(payload).__name__}, "events": []}
    return {"ok": True, "status": status, "events": payload, "params": params, "url": url}

def normalize_game_lines(events: list) -> pd.DataFrame:
    rows = []
    for ev in events or []:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}"
        commence = ev.get("commence_time")

        for book in ev.get("bookmakers", []) or []:
            bkey = book.get("key")
            btitle = book.get("title", bkey)
            for m in book.get("markets", []) or []:
                mkey = m.get("key")
                for out in m.get("outcomes", []) or []:
                    name = out.get("name")
                    price = out.get("price")
                    point = out.get("point", np.nan)

                    if mkey == "h2h":
                        bet_type = "Moneyline"
                        side = name
                        line = np.nan
                    elif mkey == "spreads":
                        bet_type = "Spread"
                        side = name
                        line = point
                    elif mkey == "totals":
                        bet_type = "Total"
                        side = name  # Over/Under
                        line = point
                    else:
                        continue

                    if price is None:
                        continue

                    rows.append({
                        "EventID": event_id,
                        "Matchup": matchup,
                        "Commence": commence,
                        "BetType": bet_type,
                        "Side": side,
                        "Line": line,
                        "Book": btitle,
                        "BookKey": bkey,
                        "Price": float(price),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Implied probs
    df["Implied"] = df["Price"].apply(american_to_implied)

    # Build a "market avg" implied (no-vig for 2-way when possible)
    # Group keys depend on bet type:
    # - Moneyline/Spread: group by matchup + bet type + line + side
    # - Total: group by matchup + bet type + line + side (Over/Under)
    grp = ["EventID", "BetType", "Line", "Side"]

    # Compute avg implied across books per selection
    avg = df.groupby(grp, dropna=False)["Implied"].mean().reset_index().rename(columns={"Implied": "AvgImplied"})

    # Merge back (robust to dtype mismatches)
    for c in ["Line"]:
        if c in df.columns and c in avg.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            avg[c] = pd.to_numeric(avg[c], errors="coerce")

    df = df.merge(avg, on=grp, how="left")

    # Best price per selection across books (absolute best odds for bettor: max american price)
    best = df.sort_values("Price", ascending=False).groupby(grp, dropna=False).head(1).copy()
    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook", "Implied": "BestImplied"})
    # Keep model prob = AvgImplied (simple consensus proxy)
    best["ModelProb"] = best["AvgImplied"]

    # EV in percentage points (not dollars): (ModelProb - BestImplied)*100
    best["EV"] = (best["ModelProb"] - best["BestImplied"]) * 100.0

    # Format time
    best["CommenceUTC"] = best["Commence"].apply(as_utc)
    return best

def plot_ev_bar(df: pd.DataFrame, title: str):
    if df.empty:
        return
    fig = plt.figure()
    labels = (df["BetType"] + " | " + df["Matchup"]).tolist()
    values = (df["EV"] / 100.0).tolist()  # percent in [0,1] space for axis formatting
    plt.barh(range(len(values)), values)
    plt.yticks(range(len(values)), labels)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# PGA (DataGolf): Win + Top10
# -----------------------------
# We will:
# 1) Pull DataGolf pre-tournament probabilities (baseline + course history & fit are included by DG)
# 2) Pull player decompositions to extract SG T2G + Putting predictions (and any course adj fields if present)
# 3) Blend into an "AdjustedProb" for win & top10 using a stable, explainable weighting

@st.cache_data(ttl=60 * 60 * 12)  # 12 hours cache
def datagolf_pre_tournament(tour: str = "pga"):
    if not DATAGOLF_API_KEY:
        return {"ok": False, "error": "Missing DATAGOLF_API_KEY", "rows": []}
    url = "https://feeds.datagolf.com/preds/pre-tournament"
    params = {
        "tour": tour,
        "odds_format": "percent",   # probabilities as % in output (DG)
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    ok, status, payload, err = safe_get(url, params=params)
    if not ok:
        return {"ok": False, "status": status, "error": err, "rows": []}
    if not isinstance(payload, (list, dict)):
        return {"ok": False, "status": status, "error": {"payload_type": type(payload).__name__}, "rows": []}

    # DG sometimes returns dict with "data" etc; handle both
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        rows = []
    return {"ok": True, "status": status, "rows": rows, "url": url, "params": params}

@st.cache_data(ttl=60 * 60 * 12)
def datagolf_player_decompositions(tour: str = "pga"):
    if not DATAGOLF_API_KEY:
        return {"ok": False, "error": "Missing DATAGOLF_API_KEY", "rows": []}
    url = "https://feeds.datagolf.com/preds/player-decompositions"
    params = {
        "tour": tour,
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    ok, status, payload, err = safe_get(url, params=params)
    if not ok:
        return {"ok": False, "status": status, "error": err, "rows": []}
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        rows = []
    return {"ok": True, "status": status, "rows": rows, "url": url, "params": params}

def _pick_prob_column(df: pd.DataFrame, target: str):
    """
    target: 'win' or 'top_10'
    DataGolf columns vary. We'll search for best candidate columns.

    We prefer columns that include course history & fit when available.
    """
    cols = list(df.columns)

    # normalize for matching
    low = {c: c.lower() for c in cols}

    # candidates containing target
    candidates = [c for c in cols if target in low[c]]

    # prefer "baseline + course history & fit" keywords if present
    prefer = []
    for c in candidates:
        lc = low[c]
        if ("history" in lc) or ("fit" in lc) or ("course" in lc):
            prefer.append(c)
    if prefer:
        # if multiple, pick the one with most "course-ish" words
        prefer = sorted(prefer, key=lambda c: sum(k in low[c] for k in ["course", "history", "fit"]), reverse=True)
        return prefer[0]

    # else fallback to any candidate
    if candidates:
        return candidates[0]

    return None

def normalize_datagolf_probs(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df

    # Identify player name column
    name_col = None
    for c in df.columns:
        if str(c).lower() in ["player_name", "name", "player"]:
            name_col = c
            break
    if not name_col:
        # last resort: look for anything with "player" substring
        for c in df.columns:
            if "player" in str(c).lower():
                name_col = c
                break

    if not name_col:
        return pd.DataFrame()

    df = df.rename(columns={name_col: "Player"})

    # Pick probability columns
    win_col = _pick_prob_column(df, "win")
    top10_col = _pick_prob_column(df, "top_10")

    # DG percent output: usually 0-100 (%). Convert to 0-1.
    out = df[["Player"]].copy()

    def to_prob01(x):
        try:
            v = float(x)
        except Exception:
            return np.nan
        # if value > 1.2 assume it's percent
        if v > 1.2:
            return v / 100.0
        return v

    if win_col:
        out["DG_WinProb"] = df[win_col].apply(to_prob01)
    else:
        out["DG_WinProb"] = np.nan

    if top10_col:
        out["DG_Top10Prob"] = df[top10_col].apply(to_prob01)
    else:
        out["DG_Top10Prob"] = np.nan

    return out.dropna(subset=["DG_WinProb", "DG_Top10Prob"], how="all")

def normalize_datagolf_decomp(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    if df.empty:
        return df

    # Identify player name
    name_col = None
    for c in df.columns:
        if str(c).lower() in ["player_name", "name", "player"]:
            name_col = c
            break
    if not name_col:
        for c in df.columns:
            if "player" in str(c).lower():
                name_col = c
                break
    if not name_col:
        return pd.DataFrame()

    df = df.rename(columns={name_col: "Player"})

    # Try to find predicted SG columns (names vary)
    def find_col(keys):
        for c in df.columns:
            lc = str(c).lower()
            if all(k in lc for k in keys):
                return c
        return None

    # Tee-to-green and putting predicted contributions
    t2g_col = find_col(["t2g"]) or find_col(["tee", "green"]) or find_col(["tee-to-green"])
    putt_col = find_col(["putt"]) or find_col(["putting"])

    # Course fit / history adjustments sometimes present
    fit_col = find_col(["fit"]) or find_col(["course", "fit"])
    hist_col = find_col(["history"]) or find_col(["course", "history"])

    out = df[["Player"]].copy()

    for label, col in [("Pred_T2G", t2g_col), ("Pred_Putt", putt_col), ("CourseFitAdj", fit_col), ("CourseHistAdj", hist_col)]:
        if col and col in df.columns:
            out[label] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[label] = np.nan

    return out

def build_pga_model(df_probs: pd.DataFrame, df_decomp: pd.DataFrame) -> pd.DataFrame:
    # Merge
    df = df_probs.merge(df_decomp, on="Player", how="left")

    # Recent form proxy:
    # Using predicted T2G + Putt already incorporates recent skill signals; we also compute a "FormIndex"
    # based on z-scored T2G and Putt.
    for c in ["Pred_T2G", "Pred_Putt", "CourseFitAdj", "CourseHistAdj"]:
        if c not in df.columns:
            df[c] = np.nan

    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        mu = np.nanmean(s)
        sd = np.nanstd(s)
        if not np.isfinite(sd) or sd == 0:
            return s * 0.0
        return (s - mu) / sd

    df["zT2G"] = zscore(df["Pred_T2G"])
    df["zPutt"] = zscore(df["Pred_Putt"])
    df["zFit"] = zscore(df["CourseFitAdj"])
    df["zHist"] = zscore(df["CourseHistAdj"])

    # Blend:
    # - DG already includes course history + fit in its own model probabilities.
    # - We gently adjust probabilities using SG signals (T2G/Putt) and explicit adj columns if present.
    # This avoids wild swings but still reflects the features you requested.
    #
    # We create an "adjust score" and then re-allocate probabilities via softmax while preserving the
    # overall distribution scale.
    df["AdjScore"] = (
        1.00 * df["zT2G"].fillna(0) +
        0.55 * df["zPutt"].fillna(0) +
        0.35 * df["zFit"].fillna(0) +
        0.25 * df["zHist"].fillna(0)
    )

    # Adjust win probs
    base_win = df["DG_WinProb"].fillna(0).to_numpy()
    base_top10 = df["DG_Top10Prob"].fillna(0).to_numpy()

    score = df["AdjScore"].fillna(0).to_numpy()

    # Softmax adjustment centered on base strength
    # temperature higher = smaller shifts
    win_weights = softmax(np.log(np.clip(base_win, 1e-12, 1.0)) + 0.25 * score, temp=1.35)
    top10_weights = softmax(np.log(np.clip(base_top10, 1e-12, 1.0)) + 0.20 * score, temp=1.25)

    # Rescale to match total probability mass (so we don't create nonsense totals)
    df["Model_WinProb"] = win_weights * float(np.nansum(base_win))
    df["Model_Top10Prob"] = top10_weights * float(np.nansum(base_top10))

    # Safety: keep within [0,1]
    df["Model_WinProb"] = df["Model_WinProb"].clip(0, 1)
    df["Model_Top10Prob"] = df["Model_Top10Prob"].clip(0, 1)

    return df

def plot_pga_probs(df: pd.DataFrame, col: str, title: str, n: int = 10):
    if df.empty or col not in df.columns:
        return
    d = df.sort_values(col, ascending=False).head(n).copy()
    fig = plt.figure()
    plt.barh(d["Player"][::-1], (d[col][::-1]).values)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------
# UI Rendering
# -----------------------------
st.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>Clean picks, best prices, and PGA modeling</div>", unsafe_allow_html=True)
st.write("")

if section == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        sport_label = st.selectbox("Sport", list(ODDS_SPORTS.keys()), index=0)
    with colB:
        topn = st.slider("Top bets by EV", 2, 10, 5)
    with colC:
        show_all = st.toggle("Show full normalized table", value=False)

    sport_key = ODDS_SPORTS[sport_label]

    if debug:
        st.code({"sport_key": sport_key, "markets": GAME_MARKETS, "bookmakers": BOOKMAKERS})

    if not ODDS_API_KEY:
        st.error("Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY.")
        st.stop()

    res = fetch_game_lines(sport_key)
    if debug:
        st.write("Debug — API status:", res.get("status"), "ok:", res.get("ok"))
        if not res.get("ok"):
            st.json(res.get("error"))

    if not res.get("ok") or not res.get("events"):
        st.warning("No game line data available right now for these settings (or no upcoming games returned).")
        st.stop()

    df_best = normalize_game_lines(res["events"])
    if df_best.empty:
        st.warning("No game line rows were normalized from the API response.")
        if debug:
            st.write("Raw sample (first event):")
            st.json(res["events"][0] if res["events"] else {})
        st.stop()

    # Top picks
    df_best = df_best.sort_values("EV", ascending=False)
    picks = df_best.head(int(topn)).copy()

    st.subheader(f"{sport_label} — Game Lines (DraftKings + FanDuel)")
    st.caption("Best bookmaker per line is already selected (BestBook/BestPrice). EV is based on consensus avg implied across books vs best implied.")

    # Display picks
    display_cols = ["BetType", "Matchup", "Side", "Line", "BestPrice", "BestBook", "ModelProb", "BestImplied", "EV"]
    for c in display_cols:
        if c not in picks.columns:
            picks[c] = np.nan

    # Format for readability
    out = picks[display_cols].copy()
    out["ModelProb"] = (out["ModelProb"] * 100).round(1)
    out["BestImplied"] = (out["BestImplied"] * 100).round(1)
    out["EV"] = out["EV"].round(2)

    st.dataframe(out, use_container_width=True, hide_index=True)

    plot_ev_bar(picks, f"{sport_label} — Top EV Bets (EV%)")

    if show_all:
        st.write("Full normalized best-per-line table:")
        df_show = df_best.copy()
        df_show["ModelProb%"] = (df_show["ModelProb"] * 100).round(2)
        df_show["Implied%"] = (df_show["BestImplied"] * 100).round(2)
        df_show["EV"] = df_show["EV"].round(2)
        st.dataframe(
            df_show[["BetType","Matchup","Side","Line","BestPrice","BestBook","ModelProb%","Implied%","EV"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "PGA (Win + Top-10)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    with colA:
        pga_tour = st.selectbox("Tour", ["pga"], index=0)
    with colB:
        topk = st.slider("Top picks per market", 3, 10, 5)

    if not DATAGOLF_API_KEY:
        st.error("Missing DATAGOLF_API_KEY. Add it in Streamlit Secrets as DATAGOLF_API_KEY.")
        st.stop()

    probs_res = datagolf_pre_tournament(tour=pga_tour)
    decomp_res = datagolf_player_decompositions(tour=pga_tour)

    if debug:
        st.write("Debug — DataGolf pre-tournament:", probs_res.get("status"), "ok:", probs_res.get("ok"))
        st.write("Debug — DataGolf decompositions:", decomp_res.get("status"), "ok:", decomp_res.get("ok"))
        if not probs_res.get("ok"):
            st.json(probs_res.get("error"))
        if not decomp_res.get("ok"):
            st.json(decomp_res.get("error"))

    if not probs_res.get("ok") or not probs_res.get("rows"):
        st.warning("No DataGolf pre-tournament rows returned right now.")
        st.stop()

    df_probs = normalize_datagolf_probs(probs_res["rows"])
    if df_probs.empty:
        st.warning("Could not normalize DataGolf probabilities (schema changed or missing columns).")
        if debug:
            st.write("Sample row:")
            st.json(probs_res["rows"][0] if probs_res["rows"] else {})
        st.stop()

    df_decomp = normalize_datagolf_decomp(decomp_res.get("rows", []))
    df_model = build_pga_model(df_probs, df_decomp)

    st.subheader("PGA — Win + Top-10 Picks (Course history + course fit + recent form + SG T2G/Putting)")

    # WIN picks
    win = df_model.dropna(subset=["Model_WinProb"]).sort_values("Model_WinProb", ascending=False).head(int(topk)).copy()
    win["Win%"] = (win["Model_WinProb"] * 100).round(2)
    win["DG_Win%"] = (win["DG_WinProb"] * 100).round(2)
    st.markdown("### Win — Top Picks")
    st.dataframe(
        win[["Player","Win%","DG_Win%","Pred_T2G","Pred_Putt","CourseFitAdj","CourseHistAdj"]],
        use_container_width=True,
        hide_index=True
    )

    # TOP10 picks
    top10 = df_model.dropna(subset=["Model_Top10Prob"]).sort_values("Model_Top10Prob", ascending=False).head(int(topk)).copy()
    top10["Top10%"] = (top10["Model_Top10Prob"] * 100).round(2)
    top10["DG_Top10%"] = (top10["DG_Top10Prob"] * 100).round(2)
    st.markdown("### Top-10 — Top Picks")
    st.dataframe(
        top10[["Player","Top10%","DG_Top10%","Pred_T2G","Pred_Putt","CourseFitAdj","CourseHistAdj"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Probability Snapshots")
    plot_pga_probs(df_model, "Model_WinProb", "Model Win Probability (Top 10)", n=10)
    plot_pga_probs(df_model, "Model_Top10Prob", "Model Top-10 Probability (Top 10)", n=10)

    # Top 25 snapshot
    st.markdown("### Top 25 Snapshot (Win + Top-10)")
    snap = df_model.copy()
    snap["Win%"] = (snap["Model_WinProb"] * 100).round(2)
    snap["Top10%"] = (snap["Model_Top10Prob"] * 100).round(2)
    snap = snap.sort_values(["Win%","Top10%"], ascending=False).head(25)
    st.dataframe(
        snap[["Player","Win%","Top10%","Pred_T2G","Pred_Putt","CourseFitAdj","CourseHistAdj"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)
