# app.py — EdgeLedger Premium (Game Lines + Player Props + PGA + UFC + Results Repo)
# NOTE:
# - Keeps your existing Game Lines / Player Props / PGA logic intact.
# - Removes Tracker mode entirely.
# - Adds a non-interactive "Results" repository that auto-snapshots top picks and grades them from live/final scores.
# - Adds a UFC tab wired in but designed to fail-safe (will not break other modules if upstream sites block/change).

import os
import time
import json
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Optional OpenAI layer (safe/optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# Page + Premium Theme (UI only)
# =========================================================
st.set_page_config(page_title="EdgeLedger", layout="wide", initial_sidebar_state="expanded")

PREMIUM_CSS = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

:root{
  --bg0:#070b14; --bg1:#0b1220; --bg2:#0f172a;
  --card:#0b1220; --stroke: rgba(148,163,184,0.18);
  --text:#e5e7eb; --muted:#94a3b8; --good:#22c55e; --bad:#ef4444;
}

html, body, [class*="css"]  { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1400px; }
@media (min-width: 1400px) { .block-container { max-width: 1700px; } }

section[data-testid="stSidebar"] {
  background: radial-gradient(1200px 800px at 10% 10%, rgba(59,130,246,0.12), transparent 50%),
              linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.big-title { font-size: 1.9rem; font-weight: 950; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.subtle { color: var(--muted); font-size: 0.95rem; margin-bottom: 0.35rem; }

.card {
  background: radial-gradient(900px 500px at 10% 0%, rgba(99,102,241,0.10), transparent 55%),
              linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.pill { display:inline-block; padding:0.20rem 0.60rem; border-radius:999px;
  background:rgba(255,255,255,0.08); margin-right:0.4rem; font-size:0.85rem; }
.small {font-size:0.85rem; color:var(--muted);}
hr { border: none; border-top: 1px solid var(--stroke); margin: 10px 0; }

div[data-testid="stDataFrame"] { width: 100%; }
div[data-testid="stDataFrame"] > div { overflow-x: auto !important; }

@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .big-title { font-size: 1.35rem; }
  .subtle { font-size: 0.85rem; }
  .card { padding: 10px 10px; border-radius: 14px; }
  .stMarkdown p, .stCaption { font-size: 0.9rem; }
  canvas, svg, img { max-width: 100% !important; height: auto !important; }
  div[role="radiogroup"] label { padding: 10px 10px !important; margin: 6px 0 !important; border-radius: 12px !important; }
  div[role="radiogroup"] label p { font-size: 1.05rem !important; font-weight: 700 !important; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# =========================================================
# Helpers
# =========================================================
def card_open():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _utc_now():
    return datetime.now(timezone.utc)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# =========================================================
# Keys (Secrets -> Env -> Session override)
# =========================================================
def get_key(name: str, default: str = "") -> str:
    if name in st.session_state and str(st.session_state[name]).strip():
        return str(st.session_state[name]).strip()
    if hasattr(st, "secrets") and name in st.secrets:
        v = str(st.secrets.get(name, "")).strip()
        if v:
            return v
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

ODDS_API_KEY = get_key("ODDS_API_KEY", "")
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "") or get_key("DATAGOLF_KEY", "")
OPENAI_API_KEY = get_key("OPENAI_API_KEY", "")

# =========================================================
# HTTP
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/2.0 (streamlit)"})

def safe_get(url: str, params: dict | None = None, timeout: int = 25):
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

def safe_get_text(url: str, timeout: int = 25):
    try:
        r = SESSION.get(url, timeout=timeout)
        ok = 200 <= r.status_code < 300
        return ok, r.status_code, r.text, r.url
    except Exception as e:
        return False, 0, str(e), url

def is_list_of_dicts(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))

# =========================================================
# Optional AI Assist (non-destructive)
# =========================================================
def ai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

@st.cache_data(ttl=60 * 15, show_spinner=False)
def ai_enrich_rows_cached(rows_json: str) -> list[dict]:
    cli = ai_client()
    if cli is None:
        return []
    rows = json.loads(rows_json)

    # Keep payload light
    prompt_rows = []
    for r in rows:
        prompt_rows.append({
            "Sport": r.get("Sport",""),
            "Market": r.get("Market",""),
            "Event": r.get("Event",""),
            "Selection": r.get("Selection",""),
            "Line": r.get("Line",""),
            "BestPrice": r.get("BestPrice",""),
            "YourProb": r.get("YourProb", None),
            "ImpliedBest": r.get("ImpliedBest", None),
            "Edge": r.get("Edge", None),
        })

    # Minimal schema-ish output without requiring special SDK features across environments
    # (returns JSON list; safe parse)
    resp = cli.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": (
                "You are a sports betting analytics assistant. "
                "For each row, return JSON with: label (short), confidence (0..1), "
                "reasoning_bullets (3-6 short), trend_flags (0-6 short). "
                "Do not hallucinate injuries/news. Use only what is in the row.\n\n"
                f"ROWS:\n{json.dumps(prompt_rows)}\n\n"
                "Return ONLY JSON: {\"items\":[...]}."
            )
        }],
    )
    try:
        out = json.loads(resp.output_text)
        return out.get("items", []) if isinstance(out, dict) else []
    except Exception:
        return []

def apply_ai_enrichment(df_best: pd.DataFrame, sport: str, market: str, max_rows: int = 25) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    rows = []
    for _, r in df_best.head(max_rows).iterrows():
        rows.append({
            "Sport": sport,
            "Market": market,
            "Event": r.get("Event",""),
            "Selection": r.get("Outcome", r.get("Selection","")),
            "Line": r.get("LineBucket", r.get("Line","")),
            "BestPrice": r.get("BestPrice",""),
            "YourProb": float(r.get("YourProb", np.nan)) if pd.notna(r.get("YourProb", np.nan)) else None,
            "ImpliedBest": float(r.get("ImpliedBest", np.nan)) if pd.notna(r.get("ImpliedBest", np.nan)) else None,
            "Edge": float(r.get("Edge", np.nan)) if pd.notna(r.get("Edge", np.nan)) else None,
        })
    enrich = ai_enrich_rows_cached(json.dumps(rows))
    out = df_best.copy()
    if not enrich:
        return out

    for c in ["AI_Label","AI_Conf","AI_Flags","AI_Notes"]:
        if c not in out.columns:
            out[c] = ""

    for i, e in enumerate(enrich[:max_rows]):
        out.loc[out.index[i], "AI_Label"] = str(e.get("label",""))
        out.loc[out.index[i], "AI_Conf"] = float(e.get("confidence", 0.0) or 0.0)
        out.loc[out.index[i], "AI_Flags"] = ", ".join(e.get("trend_flags", []) or [])
        out.loc[out.index[i], "AI_Notes"] = " • ".join(e.get("reasoning_bullets", []) or [])
    return out

# =========================================================
# Odds math (unchanged)
# =========================================================
def american_to_implied(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def clamp01(x):
    return np.clip(x, 0.001, 0.999)

def pct01_to_100(series01):
    return (pd.to_numeric(series01, errors="coerce") * 100.0).round(2)

def line_bucket_half_point(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    return round(v * 2.0) / 2.0

# =========================================================
# API Config (The Odds API)
# =========================================================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # ONLY these

SPORT_KEYS_LINES = {
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
PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Edge = YourProb − ImpliedProb(best price)</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
show_non_value = st.sidebar.checkbox("Show non-value rows (Edge ≤ 0)", value=False)

mode = st.sidebar.radio(
    "Mode",
    ["Game Lines", "Player Props", "PGA", "UFC", "Results"],
    index=0
)

with st.sidebar.expander("API Keys (session-only override)", expanded=False):
    st.caption("If Secrets aren’t set, paste keys here (session-only).")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY / DATAGOLF_API_KEY", value=DATAGOLF_API_KEY or "", type="password")
    oa_in = st.text_input("OPENAI_API_KEY (optional)", value=OPENAI_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()
    if oa_in.strip():
        st.session_state["OPENAI_API_KEY"] = oa_in.strip()
        OPENAI_API_KEY = oa_in.strip()

with st.sidebar.expander("AI Assist (optional)", expanded=False):
    ai_on = st.toggle("Enable AI Assist (labels + notes)", value=False)
    st.caption("AI Assist never replaces your math. It only adds short labels/notes.")

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Ranked by **Edge = YourProb − ImpliedProb(best price)**. "
    "Modules run independently (no cross-impact). "
    "Results repo auto-snapshots top picks + auto-grades from scores."
)

if not ODDS_API_KEY.strip() and mode in ["Game Lines", "Player Props", "Results"]:
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar.')
    st.stop()

# =========================================================
# Caching (daily)
# =========================================================
@st.cache_data(ttl=60 * 60 * 24)
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

@st.cache_data(ttl=60 * 60 * 24)
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 24)
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

@st.cache_data(ttl=60 * 10)
def fetch_scores(sport_key: str, days_from: int = 7):
    # v4 scores endpoint
    url = f"{ODDS_HOST}/sports/{sport_key}/scores"
    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": int(days_from),
    }
    ok, status, payload, final_url = safe_get(url, params=params, timeout=30)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

# =========================================================
# Normalizers (unchanged)
# =========================================================
def normalize_lines(raw):
    rows = []
    if not is_list_of_dicts(raw):
        return pd.DataFrame()
    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}"
        commence = ev.get("commence_time")
        for bm in (ev.get("bookmakers", []) or []):
            book = bm.get("title") or bm.get("key")
            for mk in (bm.get("markets", []) or []):
                mkey = mk.get("key")
                for out in (mk.get("outcomes", []) or []):
                    line = out.get("point")
                    rows.append({
                        "Scope": "GameLine",
                        "Event": matchup,
                        "Commence": commence,
                        "Market": mkey,
                        "Outcome": out.get("name"),
                        "Line": line,
                        "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                        "Price": out.get("price"),
                        "Book": book,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Outcome", "Price"])

def normalize_props(event_payload):
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    matchup = f"{away} @ {home}"

    for bm in (event_payload.get("bookmakers", []) or []):
        book = bm.get("title") or bm.get("key")
        for mk in (bm.get("markets", []) or []):
            mkey = mk.get("key")
            for out in (mk.get("outcomes", []) or []):
                player = out.get("name")
                side = out.get("description")  # Over/Under usually; may be blank
                line = out.get("point")
                price = out.get("price")
                if player is None or price is None:
                    continue
                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": str(side) if side is not None else "",
                    "Line": line,
                    "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                    "Price": price,
                    "Book": book
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Player", "Price"])

# =========================================================
# Core Best-Bet Logic (unchanged)
# =========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    return out

def compute_no_vig_within_book_two_way(df: pd.DataFrame, group_cols_book: list) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    sums = out.groupby(group_cols_book)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out

def estimate_your_prob(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
    """
    YourProb:
    - two-way: no-vig within each book, avg across books
    - one-way: market-consensus avg implied across books (enables value by shopping)
    """
    if df.empty:
        return df.copy()

    out = add_implied(df)
    side_col = "Outcome" if "Outcome" in out.columns else "Side"
    n_sides = out.groupby(key_cols)[side_col].transform(lambda s: s.nunique(dropna=True))
    out["_n_sides"] = pd.to_numeric(n_sides, errors="coerce").fillna(1)

    out = compute_no_vig_within_book_two_way(out, group_cols_book=book_cols)

    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")

    out["YourProb"] = np.where(out["_n_sides"] >= 2, nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    out = out.drop(columns=["_n_sides"], errors="ignore")
    return out

def best_price_and_edge(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Price"])

    idx = out.groupby(group_cols_best)["Price"].idxmax()
    best = out.loc[idx].copy()

    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook"})
    best["ImpliedBest"] = best["BestPrice"].apply(american_to_implied)
    best["ImpliedBest"] = clamp01(pd.to_numeric(best["ImpliedBest"], errors="coerce").fillna(0.5))

    best["Edge"] = (pd.to_numeric(best["YourProb"], errors="coerce") - pd.to_numeric(best["ImpliedBest"], errors="coerce"))
    best["EV"] = (best["Edge"] * 100.0)
    return best

def strict_no_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    STRICT contradiction removal:
    Keep exactly ONE row per contradiction group (max Edge), ignoring line differences.
    """
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    return out.loc[idx].sort_values("Edge", ascending=False)

def filter_value(df_best: pd.DataFrame, show_non_value: bool) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    out = out.sort_values("Edge", ascending=False)
    if show_non_value:
        return out
    return out[out["Edge"] > 0]

def decorate_probs(df_best: pd.DataFrame) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["YourProb%"] = pct01_to_100(out["YourProb"])
    out["Implied%"] = pct01_to_100(out["ImpliedBest"])
    out["Edge%"] = pct01_to_100(out["Edge"])
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").round(2)
    return out

# =========================================================
# Boards (unchanged)
# =========================================================
def build_game_lines_board(sport: str, bet_type: str):
    sport_key = SPORT_KEYS_LINES[sport]
    market_key = GAME_MARKETS[bet_type]

    res = fetch_game_lines(sport_key)
    if debug:
        st.json({"endpoint": "odds(game_lines)", "status": res["status"], "url": res["url"], "params": res["params"]})

    if not res["ok"] or not is_list_of_dicts(res["payload"]):
        return pd.DataFrame(), {"error": "Game lines API failed", "payload": res["payload"], "status": res["status"]}

    df = normalize_lines(res["payload"])
    if df.empty:
        return pd.DataFrame(), {"error": "No normalized game lines"}

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        return pd.DataFrame(), {"error": "No rows for selected market"}

    key_cols = ["Event", "Market", "Outcome"]
    book_cols = ["Event", "Market", "Book"]
    best_cols = ["Event", "Market", "Outcome"]

    if market_key in ["spreads", "totals"]:
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    contradiction_cols = ["Event", "Market"]
    df_best = strict_no_contradictions(df_best, contradiction_cols=contradiction_cols)

    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)
    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}

def build_props_board(sport: str, prop_label: str, max_events_scan: int = 8):
    sport_key = SPORT_KEYS_PROPS[sport]
    market_key = PROP_MARKETS[prop_label]

    ev_res = fetch_events(sport_key)
    if debug:
        st.json({"endpoint": "events", "status": ev_res["status"], "url": ev_res["url"], "params": ev_res["params"]})

    if not ev_res["ok"] or not is_list_of_dicts(ev_res["payload"]):
        return pd.DataFrame(), {"error": "Events API failed", "payload": ev_res["payload"], "status": ev_res["status"]}

    event_ids = [e.get("id") for e in ev_res["payload"] if isinstance(e, dict) and e.get("id")]
    event_ids = event_ids[: int(max_events_scan)]
    if not event_ids:
        return pd.DataFrame(), {"error": "No upcoming events"}

    all_rows = []
    call_log = []
    for eid in event_ids:
        r = fetch_event_odds(sport_key, eid, market_key)
        call_log.append({"event_id": eid, "market": market_key, "status": r["status"], "ok": r["ok"]})
        if not r["ok"] or not isinstance(r["payload"], dict):
            continue
        dfp = normalize_props(r["payload"])
        if not dfp.empty:
            all_rows.append(dfp)
        time.sleep(0.04)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {"error": "No props returned for DK/FD on scanned events (or market not posted yet).", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    best_cols = ["Event", "Market", "Player", "Side"]

    if "LineBucket" in df.columns and df["LineBucket"].notna().any():
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    contradiction_cols = ["Event", "Market", "Player"]
    df_best = strict_no_contradictions(df_best, contradiction_cols=contradiction_cols)

    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)
    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}

# =========================================================
# Charts
# =========================================================
def bar_prob(df, label_col, prob_col_percent, title):
    if df.empty:
        return
    fig = plt.figure()
    vals = pd.to_numeric(df[prob_col_percent], errors="coerce").fillna(0.0).values
    labs = df[label_col].astype(str).values
    plt.barh(labs, vals)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(100))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# PGA — Advanced DataGolf Module (unchanged)
# =========================================================
DG_HOST = "https://feeds.datagolf.com"

def _dg_get(path: str, params: dict):
    url = f"{DG_HOST}{path}"
    ok, status, payload, final_url = safe_get(url, params=params, timeout=30)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def _dg_find_rows(payload):
    if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
        return payload, {}

    if not isinstance(payload, dict):
        return [], {}

    meta = {}
    for k in ["event_name", "last_updated", "models_available", "tour"]:
        if k in payload:
            meta[k] = payload.get(k)

    for key in ["data", "predictions", "preds", "players", "rows"]:
        if key in payload and isinstance(payload[key], list):
            rows = payload[key]
            if len(rows) == 0 or isinstance(rows[0], dict):
                return rows, meta

    prefer = ["baseline_history_fit", "baseline", "sg_total", "default"]
    for p in prefer:
        if p in payload and isinstance(payload[p], list):
            rows = payload[p]
            if len(rows) == 0 or isinstance(rows[0], dict):
                meta["model_used"] = p
                return rows, meta

    for k, v in payload.items():
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            meta["model_used"] = k
            return v, meta

    return [], meta

def _normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split()).lower()

def _first_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_pga_board():
    if not DATAGOLF_API_KEY.strip():
        return {}, {"error": 'Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY).'}

    pre_params = {
        "tour": "pga",
        "add_position": 10,
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    decomp_params = {"tour": "pga", "file_format": "json", "key": DATAGOLF_API_KEY}
    skill_params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}

    pre = _dg_get("/preds/pre-tournament", pre_params)
    dec = _dg_get("/preds/player-decompositions", decomp_params)
    skl = _dg_get("/preds/skill-ratings", skill_params)

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]},
            "dg_decomp": {"ok": dec["ok"], "status": dec["status"], "url": dec["url"]},
            "dg_skill": {"ok": skl["ok"], "status": skl["status"], "url": skl["url"]},
        })

    if not pre["ok"]:
        return {}, {"error": "DataGolf pre-tournament call failed", "status": pre["status"], "payload": pre["payload"]}

    pre_rows, meta = _dg_find_rows(pre["payload"])
    if not pre_rows:
        return {}, {"error": "No PGA prediction rows returned from DataGolf.", "dg_meta": meta}

    df_pre = pd.DataFrame(pre_rows)
    name_col = _first_col(df_pre, ["player_name", "name", "golfer", "player"])
    if not name_col:
        return {}, {"error": "Could not find player name column in DataGolf payload."}

    win_col = None
    top10_col = None
    for c in df_pre.columns:
        lc = str(c).lower()
        if win_col is None and "win" in lc:
            win_col = c
        if top10_col is None and ("top" in lc and "10" in lc):
            top10_col = c

    if not win_col:
        return {}, {"error": "Could not locate win probability column."}

    df_pre["Player"] = df_pre[name_col].astype(str)
    df_pre["PlayerKey"] = df_pre["Player"].apply(_normalize_name)
    df_pre["WinProb"] = pd.to_numeric(df_pre[win_col], errors="coerce") / 100.0
    df_pre["Top10Prob"] = pd.to_numeric(df_pre[top10_col], errors="coerce") / 100.0 if top10_col else np.nan

    df_skill = pd.DataFrame()
    if skl["ok"]:
        rows_s, _ = _dg_find_rows(skl["payload"])
        if rows_s:
            df_skill = pd.DataFrame(rows_s)
            nm = _first_col(df_skill, ["player_name", "name", "player"])
            if nm:
                df_skill["PlayerKey"] = df_skill[nm].astype(str).apply(_normalize_name)

    df_dec = pd.DataFrame()
    if dec["ok"]:
        rows_d, _ = _dg_find_rows(dec["payload"])
        if rows_d:
            df_dec = pd.DataFrame(rows_d)
            nm = _first_col(df_dec, ["player_name", "name", "player"])
            if nm:
                df_dec["PlayerKey"] = df_dec[nm].astype(str).apply(_normalize_name)

    df = df_pre[["Player", "PlayerKey", "WinProb", "Top10Prob"]].copy()

    if not df_skill.empty:
        rating_col = _first_col(df_skill, ["skill", "rating", "sg_total", "value"])
        if rating_col:
            tmp = df_skill[["PlayerKey", rating_col]].copy().rename(columns={rating_col: "SkillRating"})
            df = df.merge(tmp, on="PlayerKey", how="left")

    if not df_dec.empty:
        col_map_candidates = {
            "SG_T2G": ["sg_t2g", "sg_tee_to_green", "t2g", "tee_to_green"],
            "SG_Putt": ["sg_putt", "sg_putting", "putt", "putting"],
            "BogeyAvoid": ["bogey_avoid", "bogey_avoidance"],
            "CourseFit": ["course_fit", "fit"],
            "CourseHistory": ["course_history", "history"],
            "RecentForm": ["recent_form", "form", "current_form", "recent"],
        }
        tmp = df_dec.copy()
        keep_cols = ["PlayerKey"]
        ren = {}
        for out_name, cands in col_map_candidates.items():
            c = _first_col(tmp, cands)
            if c:
                keep_cols.append(c)
                ren[c] = out_name
        if len(keep_cols) > 1:
            tmp = tmp[keep_cols].rename(columns=ren)
            df = df.merge(tmp, on="PlayerKey", how="left")

    for c in ["SkillRating", "SG_T2G", "SG_Putt", "BogeyAvoid", "CourseFit", "CourseHistory", "RecentForm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def z(x):
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return pd.Series(np.zeros(len(x)), index=x.index)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sd

    df["z_win"] = z(df["WinProb"])
    df["z_top10"] = z(df["Top10Prob"]) if df["Top10Prob"].notna().any() else 0.0
    df["z_skill"] = z(df["SkillRating"]) if "SkillRating" in df.columns else 0.0
    df["z_t2g"] = z(df["SG_T2G"]) if "SG_T2G" in df.columns else 0.0
    df["z_putt"] = z(df["SG_Putt"]) if "SG_Putt" in df.columns else 0.0
    df["z_bogey"] = z(df["BogeyAvoid"]) if "BogeyAvoid" in df.columns else 0.0
    df["z_fit"] = z(df["CourseFit"]) if "CourseFit" in df.columns else 0.0
    df["z_hist"] = z(df["CourseHistory"]) if "CourseHistory" in df.columns else 0.0
    df["z_form"] = z(df["RecentForm"]) if "RecentForm" in df.columns else 0.0

    df["WinScore"] = (
        0.55 * df["z_win"] +
        0.12 * df["z_t2g"] +
        0.08 * df["z_putt"] +
        0.08 * df["z_skill"] +
        0.06 * df["z_fit"] +
        0.06 * df["z_hist"] +
        0.05 * df["z_bogey"]
    )
    df["Top10Score"] = (
        0.55 * df["z_top10"] +
        0.12 * df["z_t2g"] +
        0.08 * df["z_putt"] +
        0.07 * df["z_skill"] +
        0.06 * df["z_fit"] +
        0.06 * df["z_hist"] +
        0.06 * df["z_bogey"]
    )
    df["OADScore"] = 0.55 * df["Top10Score"] + 0.25 * df["WinScore"] + 0.20 * df["z_t2g"]

    df["Win%"] = pct01_to_100(df["WinProb"])
    df["Top10%"] = pct01_to_100(df["Top10Prob"]) if df["Top10Prob"].notna().any() else np.nan

    winners = df.sort_values("WinScore", ascending=False).head(10).copy()
    top10s = df.sort_values("Top10Score", ascending=False).head(10).copy()
    oad = df.sort_values("OADScore", ascending=False).head(7).copy()

    return {"winners": winners, "top10s": top10s, "oad": oad, "meta": meta}, {}

# =========================================================
# Results Repository (NEW, no interaction)
# =========================================================
RESULTS_FILE = "results_repo.json"

def _load_results_repo() -> dict:
    if "results_repo" in st.session_state and isinstance(st.session_state["results_repo"], dict):
        return json.loads(json.dumps(st.session_state["results_repo"]))
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                repo = json.load(f)
        except Exception:
            repo = {}
    else:
        repo = {}
    if not isinstance(repo, dict):
        repo = {}
    if "snapshots" not in repo:
        repo["snapshots"] = []
    st.session_state["results_repo"] = json.loads(json.dumps(repo))
    return repo

def _save_results_repo(repo: dict):
    st.session_state["results_repo"] = json.loads(json.dumps(repo))
    try:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(repo, f, indent=2)
    except Exception:
        pass

def _today_key_local():
    # Local day key to bucket snapshots (simple)
    return datetime.now().strftime("%Y-%m-%d")

def _snapshot_exists(repo: dict, day_key: str, sport: str) -> bool:
    for s in repo.get("snapshots", []):
        if s.get("day") == day_key and s.get("sport") == sport and s.get("type") == "top10_ml_spreads":
            return True
    return False

def _make_top10_snapshot_for_sport(sport: str) -> dict | None:
    # top10 Moneyline + top10 Spreads (value-filtered already unless show_non_value)
    # If no rows, returns None
    out = {"day": _today_key_local(), "sport": sport, "type": "top10_ml_spreads", "created_at": datetime.now().isoformat(), "items": []}

    # Moneyline
    df_ml, err1 = build_game_lines_board(sport, "Moneyline")
    if df_ml is not None and not df_ml.empty:
        take = df_ml.head(10).copy()
        for _, r in take.iterrows():
            out["items"].append({
                "sport": sport,
                "bet_type": "Moneyline",
                "market_key": "h2h",
                "event": r.get("Event",""),
                "selection": r.get("Outcome",""),
                "line": None,
                "line_bucket": None,
                "best_price": r.get("BestPrice", None),
                "best_book": r.get("BestBook", ""),
                "your_prob": float(r.get("YourProb", np.nan)) if pd.notna(r.get("YourProb", np.nan)) else None,
                "implied_best": float(r.get("ImpliedBest", np.nan)) if pd.notna(r.get("ImpliedBest", np.nan)) else None,
                "edge": float(r.get("Edge", np.nan)) if pd.notna(r.get("Edge", np.nan)) else None,
                "status": "Pending",
                "result": "",
                "graded_at": None,
            })

    # Spreads
    df_sp, err2 = build_game_lines_board(sport, "Spreads")
    if df_sp is not None and not df_sp.empty:
        take = df_sp.head(10).copy()
        for _, r in take.iterrows():
            out["items"].append({
                "sport": sport,
                "bet_type": "Spreads",
                "market_key": "spreads",
                "event": r.get("Event",""),
                "selection": r.get("Outcome",""),
                "line": float(r.get("LineBucket", np.nan)) if pd.notna(r.get("LineBucket", np.nan)) else None,
                "line_bucket": float(r.get("LineBucket", np.nan)) if pd.notna(r.get("LineBucket", np.nan)) else None,
                "best_price": r.get("BestPrice", None),
                "best_book": r.get("BestBook", ""),
                "your_prob": float(r.get("YourProb", np.nan)) if pd.notna(r.get("YourProb", np.nan)) else None,
                "implied_best": float(r.get("ImpliedBest", np.nan)) if pd.notna(r.get("ImpliedBest", np.nan)) else None,
                "edge": float(r.get("Edge", np.nan)) if pd.notna(r.get("Edge", np.nan)) else None,
                "status": "Pending",
                "result": "",
                "graded_at": None,
            })

    if len(out["items"]) == 0:
        return None
    return out

def _parse_event_teams(event_str: str):
    # event format: "Away @ Home"
    if not isinstance(event_str, str):
        return None, None
    if " @ " in event_str:
        away, home = event_str.split(" @ ", 1)
        return away.strip(), home.strip()
    return None, None

def _grade_moneyline(selection: str, home: str, away: str, home_score: int, away_score: int):
    if any(x is None for x in [selection, home, away, home_score, away_score]):
        return None
    winner = home if home_score > away_score else away if away_score > home_score else "TIE"
    if winner == "TIE":
        return "P"
    return "W" if str(selection).strip() == str(winner).strip() else "L"

def _grade_spread(selection: str, home: str, away: str, home_score: int, away_score: int, line: float):
    # line is the point for that selection (Odds API: outcome point is the spread for that team)
    # If you bet selection with spread line, you win if (team_score + line) > opponent_score, push if ==, lose if <
    if any(x is None for x in [selection, home, away, home_score, away_score, line]):
        return None
    sel = str(selection).strip()
    if sel == str(home).strip():
        adj = home_score + float(line)
        opp = away_score
    elif sel == str(away).strip():
        adj = away_score + float(line)
        opp = home_score
    else:
        return None
    if abs(adj - opp) < 1e-9:
        return "P"
    return "W" if adj > opp else "L"

def _scores_map_by_matchup(payload):
    """
    Map: "Away @ Home" -> dict(home_score, away_score, completed, commence_time)
    The Odds API scores payload varies; handle defensively.
    """
    mp = {}
    if not is_list_of_dicts(payload):
        return mp
    for ev in payload:
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}"
        completed = bool(ev.get("completed", False))
        commence = ev.get("commence_time")
        home_score = None
        away_score = None
        # scores list could be [{name, score}, ...]
        scores = ev.get("scores")
        if isinstance(scores, list):
            for s in scores:
                if not isinstance(s, dict):
                    continue
                nm = s.get("name")
                sc = s.get("score")
                try:
                    sc_i = int(sc)
                except Exception:
                    sc_i = None
                if nm == home:
                    home_score = sc_i
                if nm == away:
                    away_score = sc_i
        mp[matchup] = {
            "home": home,
            "away": away,
            "home_score": home_score,
            "away_score": away_score,
            "completed": completed,
            "commence_time": commence,
        }
    return mp

def grade_repo_from_scores(repo: dict, days_from: int = 14):
    # Grade any Pending items where scores are completed and have scores.
    checked = 0
    graded = 0
    notes = []
    snaps = repo.get("snapshots", [])
    if not isinstance(snaps, list) or len(snaps) == 0:
        return repo, {"graded": 0, "checked": 0, "notes": ["No snapshots."]}

    # Build scores cache per sport_key used
    for sport, sport_key in SPORT_KEYS_LINES.items():
        # only if there are pending items for that sport
        has_pending = False
        for s in snaps:
            if s.get("sport") != sport:
                continue
            for it in s.get("items", []):
                if it.get("status") == "Pending":
                    has_pending = True
                    break
            if has_pending:
                break
        if not has_pending:
            continue

        sc = fetch_scores(sport_key, days_from=days_from)
        if debug:
            st.json({"scores": {"sport": sport, "status": sc["status"], "url": sc["url"]}})
        if not sc["ok"]:
            notes.append(f"Scores fetch failed for {sport}: {sc['status']}")
            continue

        mp = _scores_map_by_matchup(sc["payload"])
        for s in snaps:
            if s.get("sport") != sport:
                continue
            for it in s.get("items", []):
                if it.get("status") != "Pending":
                    continue
                checked += 1
                ev = it.get("event")
                m = mp.get(ev)
                if not m or not m.get("completed"):
                    continue
                hs, as_ = m.get("home_score"), m.get("away_score")
                if hs is None or as_ is None:
                    continue
                bet_type = it.get("bet_type")
                sel = it.get("selection")
                home = m.get("home")
                away = m.get("away")
                if bet_type == "Moneyline":
                    r = _grade_moneyline(sel, home, away, hs, as_)
                elif bet_type == "Spreads":
                    r = _grade_spread(sel, home, away, hs, as_, it.get("line_bucket"))
                else:
                    r = None
                if r in ["W", "L", "P"]:
                    it["result"] = r
                    it["status"] = "Graded"
                    it["graded_at"] = datetime.now().isoformat()
                    graded += 1

    repo["snapshots"] = snaps
    return repo, {"graded": graded, "checked": checked, "notes": notes}

def repo_items_df(repo: dict) -> pd.DataFrame:
    rows = []
    for s in repo.get("snapshots", []) or []:
        day = s.get("day")
        sport = s.get("sport")
        for it in s.get("items", []) or []:
            rows.append({
                "Day": day,
                "Sport": sport,
                "BetType": it.get("bet_type"),
                "Event": it.get("event"),
                "Selection": it.get("selection"),
                "Line": it.get("line_bucket"),
                "BestPrice": it.get("best_price"),
                "BestBook": it.get("best_book"),
                "YourProb": it.get("your_prob"),
                "ImpliedBest": it.get("implied_best"),
                "Edge": it.get("edge"),
                "Status": it.get("status"),
                "Result": it.get("result"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # normalize dates
    df["Day_dt"] = pd.to_datetime(df["Day"], errors="coerce")
    return df

def summarize_repo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Window","BetType","Picks","Graded","Wins","Losses","Pushes","Win%"])
    now = pd.Timestamp.now()
    today = now.normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    def window_df(label, start):
        x = df[df["Day_dt"] >= start].copy()
        if x.empty:
            return None
        x["Picks"] = 1
        x["Graded"] = (x["Status"] == "Graded").astype(int)
        x["Wins"] = (x["Result"] == "W").astype(int)
        x["Losses"] = (x["Result"] == "L").astype(int)
        x["Pushes"] = (x["Result"] == "P").astype(int)
        agg = x.groupby(["BetType"], dropna=False).agg(
            Picks=("Picks","sum"),
            Graded=("Graded","sum"),
            Wins=("Wins","sum"),
            Losses=("Losses","sum"),
            Pushes=("Pushes","sum"),
        ).reset_index()
        denom = (agg["Wins"] + agg["Losses"]).replace(0, np.nan)
        agg["Win%"] = ((agg["Wins"] / denom) * 100.0).round(1).fillna(0.0)
        agg.insert(0, "Window", label)
        return agg

    parts = []
    for label, start in [("Today", today), ("This Week", week_start), ("This Month", month_start)]:
        p = window_df(label, start)
        if p is not None:
            parts.append(p)
    if not parts:
        return pd.DataFrame(columns=["Window","BetType","Picks","Graded","Wins","Losses","Pushes","Win%"])
    return pd.concat(parts, ignore_index=True)

# =========================================================
# Team Context (best-effort, safe; does not affect pick math)
# Uses ESPN public endpoints when available.
# =========================================================
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

@st.cache_data(ttl=60 * 60 * 6)
def espn_team_records_nfl():
    # standings endpoint can change; keep defensive
    url = f"{ESPN}/football/nfl/standings"
    ok, status, payload, final_url = safe_get(url, params=None, timeout=25)
    if not ok or not isinstance(payload, dict):
        return {}
    # attempt parse: payload["children"][...]["standings"]["entries"]
    out = {}
    try:
        for conf in payload.get("children", []) or []:
            for div in conf.get("children", []) or []:
                for e in (div.get("standings", {}) or {}).get("entries", []) or []:
                    team = (((e.get("team") or {}).get("displayName")) or "").strip()
                    if not team:
                        continue
                    # records list contains items with "type": "total", "home", "away" etc
                    recs = {}
                    for r in e.get("stats", []) or []:
                        # stats list often includes "name":"overall","value": etc — not stable
                        pass
                    # Better: use e["records"] if present (often present in other ESPN schemas)
                    for rr in e.get("records", []) or []:
                        rtype = rr.get("type")
                        summ = rr.get("summary")
                        if rtype and summ:
                            recs[rtype] = summ
                    out[team] = recs
    except Exception:
        return out
    return out

def _team_context_row(event_str: str, sport: str):
    # Currently best-effort only for NFL via ESPN standings
    away, home = _parse_event_teams(event_str)
    if sport != "NFL":
        return {}
    recs = espn_team_records_nfl()
    return {
        "AwayTeam": away,
        "HomeTeam": home,
        "AwayRec": (recs.get(away, {}) or {}).get("total", ""),
        "HomeRec": (recs.get(home, {}) or {}).get("total", ""),
        "AwayHomeSplit": (recs.get(away, {}) or {}).get("home", ""),
        "AwayAwaySplit": (recs.get(away, {}) or {}).get("away", ""),
        "HomeHomeSplit": (recs.get(home, {}) or {}).get("home", ""),
        "HomeAwaySplit": (recs.get(home, {}) or {}).get("away", ""),
        # ATS is not reliably available from ESPN standings; leave blank (avoid false data)
        "AwayATS": "",
        "HomeATS": "",
        "AwayRank": "",
        "HomeRank": "",
        "AwayL5_SU": "",
        "HomeL5_SU": "",
    }

# =========================================================
# UFC Module (fail-safe; never blocks app)
# Uses UFCStats first; if blocked, shows clean message.
# =========================================================
UFCSTATS_EVENTS = "http://ufcstats.com/statistics/events/upcoming"

def ufc_render():
    card_open()
    st.subheader("UFC — Fight Picks (best-effort)")
    st.caption("This module is isolated: if UFCStats is blocked/changes, it won’t affect other modules.")

    ok, status, html, final_url = safe_get_text(UFCSTATS_EVENTS, timeout=20)
    if not ok or not isinstance(html, str) or len(html) < 200:
        st.warning("Could not load UFC upcoming events (UFCStats blocked or down).")
        if debug:
            st.json({"ufc_upcoming": {"ok": ok, "status": status, "url": final_url}})
        card_close()
        return

    # No lxml usage; very simple parsing with string ops (robust enough to not crash)
    # Find first event-details link
    # UFCStats links look like: href="http://ufcstats.com/event-details/...."
    ev_urls = []
    parts = html.split('href="')
    for p in parts[1:]:
        u = p.split('"', 1)[0]
        if "ufcstats.com/event-details/" in u:
            ev_urls.append(u)
    ev_urls = list(dict.fromkeys(ev_urls))  # unique preserve order

    if not ev_urls:
        st.warning("No UFC events found from UFCStats right now (site may be blocking or markup changed).")
        if debug:
            st.json({"hint": "No event-details links parsed from upcoming page."})
        card_close()
        return

    selected = st.selectbox("Upcoming UFC events (UFCStats)", ev_urls, index=0, key="ufc_ev")
    ok2, status2, ev_html, ev_url = safe_get_text(selected, timeout=20)
    if not ok2 or not isinstance(ev_html, str) or len(ev_html) < 200:
        st.warning("Could not load selected UFC event page.")
        if debug:
            st.json({"selected_event": {"ok": ok2, "status": status2, "url": ev_url}})
        card_close()
        return

    # Fight rows have links to fight-details
    fight_urls = []
    for p in ev_html.split('href="')[1:]:
        u = p.split('"', 1)[0]
        if "ufcstats.com/fight-details/" in u:
            fight_urls.append(u)
    fight_urls = list(dict.fromkeys(fight_urls))

    if not fight_urls:
        st.warning("No fights parsed on this event page (HTML changed or blocked).")
        if debug:
            st.json({"selected_event_url": selected, "event_url": ev_url})
        card_close()
        return

    # Minimal “best effort” picks: without reliable fighter stats (blocked scraping), we cannot compute your full metric model.
    # So we show fight list and a neutral placeholder pick until a stable stats source is wired.
    st.markdown("### Card (parsed)")
    st.caption("If you want the full metric model (age/reach/stance/record + striking/takedowns), UFCStats must allow fighter pages. This app will not scrape aggressively (keeps site load safe).")

    # Try parse fighter names from event HTML table rows
    # UFCStats event table contains <td class="b-fight-details__table-col l-page_align_left"> with names inside <a>
    names = []
    for seg in ev_html.split('b-link b-link_style_black'):
        if 'href="http://ufcstats.com/fighter-details/' in seg:
            # name text after >
            t = seg.split(">", 1)
            if len(t) > 1:
                nm = t[1].split("<", 1)[0].strip()
                if nm:
                    names.append(nm)
    # crude pairing (names appear twice per fight)
    fights = []
    if len(names) >= 2:
        for i in range(0, len(names) - 1, 2):
            fights.append((names[i], names[i+1]))

    if not fights:
        st.info("Fight list found, but fighter names could not be parsed reliably from this page. Debug on for diagnostics.")
        if debug:
            st.json({"fight_urls_count": len(fight_urls)})
        card_close()
        return

    rows = []
    for a, b in fights[:15]:
        # Placeholder heuristic: pick alphabetically earlier as “Pick” (NOT a real model)
        # This is intentional to avoid fake “AI certainty” until stable stats source is available.
        pick = a
        rows.append({"Fighter A": a, "Fighter B": b, "Pick (placeholder)": pick, "Confidence": 0.50, "Notes": "Needs stable stats feed to compute model."})

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if debug:
        st.json({"ufc_event_url": selected, "fights_parsed": len(fights)})

    card_close()

# =========================================================
# MAIN UI
# =========================================================
if mode == "Game Lines":
    card_open()

    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_bettype")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="gl_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_top25")
    show_team_context = st.toggle("Show team context (best-effort)", value=True, key="gl_team_ctx")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No game lines available."))
        card_close()
        st.stop()

    if ai_on:
        df_best = apply_ai_enrichment(df_best, sport=sport, market=bet_type, max_rows=25)

    st.subheader(f"{sport} — {bet_type} (DK/FD) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per game per market. Ranked by Edge. (Team context is additive only.)")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Outcome"]
    if "LineBucket" in top.columns and top["LineBucket"].notna().any():
        cols += ["LineBucket"]
    cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    if ai_on:
        cols += ["AI_Label", "AI_Conf", "AI_Flags", "AI_Notes"]
    cols = [c for c in cols if c in top.columns]

    if show_team_context:
        ctx_rows = []
        for ev in top["Event"].astype(str).tolist():
            ctx_rows.append(_team_context_row(ev, sport))
        ctx = pd.DataFrame(ctx_rows)
        if not ctx.empty:
            # join by index order
            top2 = top.reset_index(drop=True).copy()
            ctx2 = ctx.reset_index(drop=True).copy()
            view = pd.concat([top2[cols], ctx2], axis=1)
            st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            st.dataframe(top[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    if "YourProb%" in chart.columns:
        bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    if "Implied%" in chart.columns:
        bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        st.dataframe(snap[cols], use_container_width=True, hide_index=True)

    card_close()

elif mode == "Player Props":
    card_open()

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_prop")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="pp_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_top25")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 14, 8, key="pp_scan")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No props returned for DK/FD on scanned events."))
        card_close()
        st.stop()

    if ai_on:
        df_best = apply_ai_enrichment(df_best, sport=sport, market=prop_label, max_rows=25)

    st.subheader(f"{sport} — Player Props ({prop_label}) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per player per market per game. Ranked by Edge. AI adds notes only.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"]
    if "LineBucket" in top.columns and top["LineBucket"].notna().any():
        cols += ["LineBucket"]
    cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    if ai_on:
        cols += ["AI_Label", "AI_Conf", "AI_Flags", "AI_Notes"]
    cols = [c for c in cols if c in top.columns]

    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    if "YourProb%" in chart.columns:
        bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    if "Implied%" in chart.columns:
        bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        st.dataframe(snap[cols], use_container_width=True, hide_index=True)

    card_close()

elif mode == "PGA":
    card_open()

    if not DATAGOLF_API_KEY.strip():
        st.warning('Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY).')
        card_close()
        st.stop()

    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")
    st.caption("Top picks for Win / Top-10 + One-and-Done using DataGolf model probabilities + SG splits + fit/history/form proxies.")

    out, err = build_pga_board()
    if isinstance(out, dict) and "winners" in out:
        winners = out["winners"]
        top10s = out["top10s"]
        oad = out["oad"]

        st.markdown("### 🏆 Best Win Picks (Top 10)")
        st.dataframe(winners, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Best Top-10 Picks (Top 10)")
        st.dataframe(top10s, use_container_width=True, hide_index=True)

        st.markdown("### 🧳 Best One-and-Done Options (Top 7)")
        st.dataframe(oad, use_container_width=True, hide_index=True)
    else:
        st.warning(err.get("error", "No PGA data available right now."))
        if debug:
            st.json(err)

    card_close()

elif mode == "UFC":
    ufc_render()

else:
    # RESULTS
    card_open()
    st.subheader("📊 Results Repository — Auto-graded Win/Loss (Daily/Weekly/Monthly)")
    st.caption("Auto-snapshots top 10 Moneyline + top 10 Spreads per sport per day. Grades automatically from scores when games complete.")

    repo = _load_results_repo()

    # Auto-snapshot each sport once per day (safe)
    created = 0
    day_key = _today_key_local()
    for sport in SPORT_KEYS_LINES.keys():
        if _snapshot_exists(repo, day_key, sport):
            continue
        snap = _make_top10_snapshot_for_sport(sport)
        if snap is not None:
            repo["snapshots"].append(snap)
            created += 1
    if created > 0:
        _save_results_repo(repo)

    # Auto-grade from scores
    repo, diag = grade_repo_from_scores(repo, days_from=14)
    _save_results_repo(repo)

    if debug:
        st.json({"repo_diag": diag})

    df = repo_items_df(repo)
    if df.empty:
        st.info("No snapshots yet (no eligible games/markets returned).")
        card_close()
        st.stop()

    summary = summarize_repo(df)
    st.markdown("### Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### Latest Picks (Top 10 ML + Top 10 Spreads per sport)")
    latest_day = df["Day_dt"].max()
    show = df[df["Day_dt"] == latest_day].copy()
    show = show.sort_values(["Sport", "BetType", "Edge"], ascending=[True, True, False])

    # Add team context columns (best-effort)
    ctx_rows = []
    for _, r in show.iterrows():
        ctx_rows.append(_team_context_row(r.get("Event",""), r.get("Sport","")))
    ctx = pd.DataFrame(ctx_rows)
    if not ctx.empty:
        show2 = show.reset_index(drop=True)
        ctx2 = ctx.reset_index(drop=True)
        view = pd.concat([show2.drop(columns=["Day_dt"], errors="ignore"), ctx2], axis=1)
    else:
        view = show.drop(columns=["Day_dt"], errors="ignore")

    st.dataframe(view, use_container_width=True, hide_index=True)

    card_close()
