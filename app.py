import os
import time
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Page + Theme
# =========================
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --bg: #0b1220;
  --card: #0f1b33;
  --muted: #8aa1c7;
  --text: #e7f0ff;
  --accent: #49a4ff;
  --good: #39d98a;
  --warn: #ffcc66;
  --bad: #ff5c77;
}
html, body, [class*="css"]  { background: var(--bg) !important; color: var(--text) !important; }
h1, h2, h3, h4 { letter-spacing: -0.02em; }
.smallmuted { color: var(--muted); font-size: 0.92rem; }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 18px 18px 10px 18px;
  margin-bottom: 14px;
}
.kpi {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 16px;
  padding: 14px;
}
.badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.10);
  color: var(--muted);
}
.badge-good { color: var(--good); border-color: rgba(57,217,138,0.35); }
.badge-warn { color: var(--warn); border-color: rgba(255,204,102,0.35); }
.badge-bad  { color: var(--bad);  border-color: rgba(255,92,119,0.35); }
hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# Secrets / Keys
# =========================
# Uses Streamlit secrets if present; otherwise falls back to provided keys (so it runs immediately).
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "") or os.getenv("ODDS_API_KEY", "") or "d1a096c07dfb711c63560fcc7495fd0d"
DATAGOLF_API_KEY = st.secrets.get("DATAGOLF_API_KEY", "") or os.getenv("DATAGOLF_API_KEY", "")  # recommend store in secrets

# =========================
# Odds API Config
# =========================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # per your request
ODDS_FORMAT = "american"

SPORT_KEYS_LINES = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}

SPORT_KEYS_PROPS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

# Correct The Odds API player prop market keys
PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_pass_yds",
    "Passing TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================
# Helpers
# =========================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_get(url: str, params: dict, timeout: int = 20):
    """
    Returns: (ok: bool, status: int, payload: object, final_url: str)
    Never throws.
    """
    try:
        r = requests.get(url, params=params, timeout=timeout)
        status = r.status_code
        final_url = r.url
        try:
            payload = r.json()
        except Exception:
            payload = {"raw_text": r.text[:1000]}
        ok = (200 <= status < 300)
        return ok, status, payload, final_url
    except Exception as e:
        return False, 0, {"error": str(e)}, url

def american_to_implied(odds):
    """
    Converts American odds to implied probability (0..1).
    """
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o == 0:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def pct_fmt(series):
    # input: float probs 0..1 -> "xx.x%"
    return (pd.to_numeric(series, errors="coerce") * 100.0).round(1).astype(str) + "%"

def is_list_of_dicts(x):
    return isinstance(x, list) and all(isinstance(i, dict) for i in x)

def normalize_matchup(ev: dict) -> str:
    away = ev.get("away_team", "")
    home = ev.get("home_team", "")
    return f"{away} @ {home}".strip(" @")

# =========================
# EV / Best Price Logic (robust)
# =========================
def compute_market_consensus_prob(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    ModelProb = market consensus implied probability (average of implied probs across books).
    """
    if df.empty:
        return df
    df = df.copy()

    if "Price" not in df.columns:
        return df

    df["Implied"] = df["Price"].apply(american_to_implied)
    df = df.dropna(subset=["Implied"])

    avg = (
        df.groupby(group_cols, dropna=False)["Implied"]
        .mean()
        .reset_index()
        .rename(columns={"Implied": "ModelProb"})
    )
    # ensure merge keys align types
    for c in group_cols:
        if c in df.columns and c in avg.columns:
            df[c] = df[c].astype(str)
            avg[c] = avg[c].astype(str)

    out = df.copy()
    for c in group_cols:
        out[c] = out[c].astype(str)

    out = out.merge(avg, on=[c for c in group_cols], how="left")
    return out

def best_price(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Picks best odds per group:
      - For positive odds: higher is better
      - For negative odds: closer to 0 is better (e.g., -105 better than -120)
    """
    if df.empty:
        return df

    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    def price_rank(o):
        o = float(o)
        if o > 0:
            return o
        # -105 should beat -120 -> rank higher
        return -abs(o)

    df["_rank"] = df["Price"].apply(price_rank)

    # normalize key types so groupby won't explode
    for c in group_cols:
        df[c] = df[c].astype(str)

    idx = df.groupby(group_cols, dropna=False)["_rank"].idxmax()
    out = df.loc[idx].copy().drop(columns=["_rank"], errors="ignore")
    out = out.rename(columns={"Price": "BestPrice", "Book": "BestBook"})
    out["BestImplied"] = out["BestPrice"].apply(american_to_implied)
    return out

def compute_ev(df_best: pd.DataFrame) -> pd.DataFrame:
    """
    EV = (ModelProb - BestImplied) * 100
    Uses implied probability from the *best* odds, per your request.
    """
    if df_best.empty:
        return df_best
    df_best = df_best.copy()
    if "ModelProb" not in df_best.columns:
        df_best["ModelProb"] = np.nan
    if "BestImplied" not in df_best.columns:
        df_best["BestImplied"] = df_best.get("BestPrice", np.nan).apply(american_to_implied)

    df_best["EV"] = (pd.to_numeric(df_best["ModelProb"], errors="coerce") - pd.to_numeric(df_best["BestImplied"], errors="coerce")) * 100.0
    return df_best

def enforce_no_contradictions(df_best: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """
    If both Over and Under exist for same player+line, keep only the higher EV.
    """
    if df_best.empty:
        return df_best

    df_best = df_best.copy()

    # Normalize keys
    for c in key_cols:
        if c in df_best.columns:
            df_best[c] = df_best[c].astype(str)

    if "Side" not in df_best.columns:
        return df_best

    df_best["EV"] = pd.to_numeric(df_best.get("EV", np.nan), errors="coerce")
    df_best = df_best.sort_values("EV", ascending=False)

    keep_idx = []
    seen = set()
    for i, r in df_best.iterrows():
        k = tuple(r.get(c, "") for c in key_cols)
        if k in seen:
            continue
        seen.add(k)
        keep_idx.append(i)

    return df_best.loc[keep_idx].copy()

# =========================
# API Calls (Separate + Cached)
# =========================
@st.cache_data(ttl=60 * 60 * 24)
def fetch_game_lines(sport_key: str):
    """
    ONE daily call for game lines markets.
    """
    url = f"{ODDS_HOST}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": "h2h,spreads,totals",
        "oddsFormat": ODDS_FORMAT,
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params, "fetched_at": now_utc_iso()}

@st.cache_data(ttl=60 * 60 * 24)
def fetch_events(sport_key: str):
    """
    Daily call for events list (cheap, used for props flow).
    """
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params, "fetched_at": now_utc_iso()}

@st.cache_data(ttl=60 * 60 * 24)
def fetch_event_props_odds(sport_key: str, event_id: str, market_key: str):
    """
    Per-event props odds call. Cached daily to control usage.
    """
    url = f"{ODDS_HOST}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": ODDS_FORMAT,
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params, "event_id": event_id}

# =========================
# Normalizers
# =========================
def normalize_game_lines(payload) -> pd.DataFrame:
    rows = []
    if not is_list_of_dicts(payload):
        return pd.DataFrame()

    for ev in payload:
        matchup = normalize_matchup(ev)
        commence = ev.get("commence_time", "")

        for bm in (ev.get("bookmakers") or []):
            book = bm.get("title") or bm.get("key") or ""
            for mk in (bm.get("markets") or []):
                mk_key = mk.get("key")
                if mk_key not in ("h2h", "spreads", "totals"):
                    continue

                for out in (mk.get("outcomes") or []):
                    name = out.get("name")
                    price = out.get("price")
                    point = out.get("point", np.nan)

                    # totals: outcomes named Over/Under
                    side = name
                    if mk_key == "h2h":
                        bet_type = "Moneyline"
                    elif mk_key == "spreads":
                        bet_type = "Spread"
                    else:
                        bet_type = "Total"

                    rows.append({
                        "Event": matchup,
                        "CommenceTime": commence,
                        "Market": mk_key,
                        "Type": bet_type,
                        "Outcome": str(side),
                        "Line": point,
                        "Price": price,
                        "Book": book,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df = df.dropna(subset=["Price"])
    return df

def normalize_player_props(event_payload: dict, market_key: str) -> pd.DataFrame:
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    matchup = normalize_matchup(event_payload)
    bookmakers = event_payload.get("bookmakers") or []
    if not isinstance(bookmakers, list):
        return pd.DataFrame()

    for bm in bookmakers:
        if not isinstance(bm, dict):
            continue
        book = bm.get("title") or bm.get("key") or ""
        markets = bm.get("markets") or []
        if not isinstance(markets, list):
            continue

        for mk in markets:
            if not isinstance(mk, dict):
                continue
            if mk.get("key") != market_key:
                continue

            outcomes = mk.get("outcomes") or []
            if not isinstance(outcomes, list):
                continue

            for out in outcomes:
                if not isinstance(out, dict):
                    continue

                player = out.get("name")
                price = out.get("price")

                # For totals-style player props:
                # description often "Over"/"Under", point is the line
                side = out.get("description") or out.get("type") or ""
                line = out.get("point", np.nan)

                if player is None or price is None:
                    continue

                rows.append({
                    "Event": matchup,
                    "Market": market_key,
                    "Player": str(player),
                    "Side": str(side),
                    "Line": line,
                    "Price": price,
                    "Book": str(book),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df = df.dropna(subset=["Price", "Player"])
    return df

# =========================
# UI
# =========================
st.sidebar.markdown("## 📈 Dashboard")
st.sidebar.markdown("<div class='smallmuted'>Game lines + props (DK/FD), ranked by EV</div>", unsafe_allow_html=True)
debug = st.sidebar.toggle("Show debug", value=False)

if ODDS_API_KEY:
    st.sidebar.markdown("<span class='badge badge-good'>ODDS_API_KEY loaded</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span class='badge badge-bad'>ODDS_API_KEY missing</span>", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Section",
    ["Game Lines", "Player Props", "PGA (DataGolf)"],
    index=0,
    label_visibility="collapsed",
)

# =========================
# GAME LINES (UNCHANGED behavior)
# =========================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("EdgeLedger — Game Lines")

    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0)
    sport_key = SPORT_KEYS_LINES[sport]

    top_n = st.slider("Auto-ranked top bets (EV)", 2, 5, 3)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    res = fetch_game_lines(sport_key)
    if debug:
        st.json({
            "endpoint": "odds(game_lines)",
            "sport_key": sport_key,
            "status": res["status"],
            "ok": res["ok"],
            "url": res["url"],
            "params": res["params"],
            "fetched_at": res["fetched_at"],
        })

    if not res["ok"]:
        st.error("Game lines API failed. This will not affect props.")
        if debug:
            st.json(res["payload"])
        st.stop()

    df_raw = normalize_game_lines(res["payload"])
    if df_raw.empty:
        st.warning("No game line rows were normalized from the API response.")
        st.stop()

    # Consensus model: average implied across books per event/market/outcome/line
    group_cols = ["Event", "Market", "Type", "Outcome", "Line"]
    df_mc = compute_market_consensus_prob(df_raw, group_cols=[c for c in group_cols])
    df_best = best_price(df_mc, group_cols=[c for c in group_cols])
    df_best = compute_ev(df_best)

    # Clean display
    df_best["ModelProb%"] = pct_fmt(df_best["ModelProb"])
    df_best["Implied%"] = pct_fmt(df_best["BestImplied"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)
    df_best = df_best.sort_values("EV", ascending=False)

    top = df_best.head(int(top_n)).copy()
    top["⭐ Best Book"] = "⭐ " + top["BestBook"].astype(str)

    st.subheader(f"{sport} — Top Bets Ranked by EV")
    st.caption("Best price across DK/FD • ModelProb = market consensus • EV uses implied prob from best odds")

    cols = ["Event", "Type", "Outcome", "Line", "BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.head(25).copy()
        snap["⭐ Best Book"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Type", "Outcome", "Line", "BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PLAYER PROPS (FIXED, DOES NOT TOUCH GAME LINES)
# =========================
elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("EdgeLedger — Player Props")

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0)
    prop_label = st.selectbox("Prop Type", list(PLAYER_PROP_MARKETS.keys()), index=0)

    top_n = st.slider("Auto-ranked top picks (EV)", 2, 5, 3)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    # Usage control: how many events to scan (each is cached daily)
    max_events = st.slider("Events to scan (usage control)", 1, 20, 10)

    sport_key = SPORT_KEYS_PROPS[sport]
    market_key = PLAYER_PROP_MARKETS[prop_label]

    if debug:
        st.write(f"DEBUG: Using sport_key={sport_key}, market_key={market_key}, bookmakers={BOOKMAKERS}")

    # Step 1: events
    ev_res = fetch_events(sport_key)
    if debug:
        st.json({
            "endpoint": "events",
            "sport_key": sport_key,
            "status": ev_res["status"],
            "ok": ev_res["ok"],
            "url": ev_res["url"],
            "fetched_at": ev_res["fetched_at"],
        })

    if not ev_res["ok"] or not is_list_of_dicts(ev_res["payload"]):
        st.error("Events API did not return a valid list. Props will not load.")
        if debug:
            st.json(ev_res["payload"])
        st.stop()

    event_ids = [e.get("id") for e in ev_res["payload"] if isinstance(e, dict) and e.get("id")]
    event_ids = event_ids[: int(max_events)]

    if not event_ids:
        st.warning("No upcoming events found for props.")
        st.stop()

    # Step 2: per-event odds for that prop market
    frames = []
    call_log = []
    for eid in event_ids:
        r = fetch_event_props_odds(sport_key, eid, market_key)
        call_log.append({
            "event_id": eid,
            "status": r["status"],
            "ok": r["ok"],
            "url": r["url"],
        })

        # 422 means market not offered for that event (very common). Skip safely.
        if not r["ok"] or not isinstance(r["payload"], dict):
            continue

        dfp = normalize_player_props(r["payload"], market_key)
        if not dfp.empty:
            frames.append(dfp)

        time.sleep(0.05)  # gentle pacing

    if debug:
        st.json({
            "events_scanned": len(event_ids),
            "events_with_rows": len(frames),
            "first_calls": call_log[:8],
        })

    if not frames:
        st.warning(
            "No props returned for DK/FD on scanned events (or this market isn’t available yet). "
            "Try another prop type or scan more events."
        )
        st.stop()

    df = pd.concat(frames, ignore_index=True)

    # Market-consensus model prob + best price
    group_cols = ["Event", "Market", "Player", "Side", "Line"]
    # convert keys to strings so merges never throw type errors
    for c in group_cols:
        df[c] = df[c].astype(str)

    df_mc = compute_market_consensus_prob(df, group_cols=group_cols)
    df_best = best_price(df_mc, group_cols=group_cols)
    df_best = compute_ev(df_best)

    # No contradictory picks (Over + Under) for same player+line: keep highest EV
    key_cols = ["Event", "Market", "Player", "Line"]
    df_best = enforce_no_contradictions(df_best, key_cols=key_cols)

    df_best["ModelProb%"] = pct_fmt(df_best["ModelProb"])
    df_best["Implied%"] = pct_fmt(df_best["BestImplied"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    df_best = df_best.sort_values("EV", ascending=False)

    top = df_best.head(int(top_n)).copy()
    top["⭐ Best Book"] = "⭐ " + top["BestBook"].astype(str)

    st.subheader(f"{sport} — {prop_label} (DK/FD)")
    st.caption("Best price across DK/FD • ModelProb = market consensus • EV uses implied probability from best odds")

    cols = ["Event", "Player", "Side", "Line", "BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.head(25).copy()
        snap["⭐ Best Book"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side", "Line", "BestPrice", "⭐ Best Book", "ModelProb%", "Implied%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PGA (DataGolf) — safe stub (won’t break if key missing)
# =========================
else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.title("EdgeLedger — PGA (DataGolf)")

    if not DATAGOLF_API_KEY:
        st.warning('Missing DATAGOLF_API_KEY. Add it in Streamlit Secrets as DATAGOLF_API_KEY="..."')
        st.stop()

    st.info("Your PGA logic can sit here without impacting game lines or props. This section is isolated.")
    st.markdown("</div>", unsafe_allow_html=True)
