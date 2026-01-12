import os
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st

# =============================
# Page + Visual polish
# =============================
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; }
      [data-testid="stSidebarNav"] { display: none; } /* hide multipage nav */
      .big-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.35rem 0; }
      .subtle { color: rgba(250,250,250,0.75); }
      .pill { display:inline-block; padding:0.18rem 0.55rem; border-radius:999px; background:rgba(255,255,255,0.08); margin-right:0.4rem; }
      .card { padding: 0.9rem 1rem; border-radius: 18px; background: rgba(255,255,255,0.06); }
      hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================
# Keys (Secrets -> env -> fallback)
# =============================
ODDS_API_KEY = None
DATAGOLF_KEY = None

if hasattr(st, "secrets"):
    ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None)
    DATAGOLF_KEY = st.secrets.get("DATAGOLF_KEY", None)

ODDS_API_KEY = ODDS_API_KEY or os.getenv("ODDS_API_KEY") or "d1a096c07dfb711c63560fcc7495fd0d"
DATAGOLF_KEY = DATAGOLF_KEY or os.getenv("DATAGOLF_KEY") or "909304744927252dd7a207f7dce4"

# =============================
# Constants
# =============================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # per your request

SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

GAME_LINE_MARKETS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

# ✅ Correct Odds API market keys for NFL/CFB props (DK/FD)
PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_pass_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

DG_BASE = "https://feeds.datagolf.com"

# =============================
# Helpers
# =============================
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_box(enabled: bool, title: str, payload):
    if not enabled:
        return
    with st.expander(f"🧪 Debug — {title}", expanded=False):
        st.write(payload)

def safe_get(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        return r
    except Exception as e:
        class _R:
            status_code = 0
            headers = {}
            def json(self): return {"error": str(e)}
            text = str(e)
        return _R()

def is_list_of_dicts(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))

def american_to_implied(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def clamp01(x):
    return np.clip(x, 0.001, 0.999)

def pct(x):
    """0-1 -> 0-100"""
    return x * 100.0

# =============================
# EV + Best price (FIXED: no merge)
# =============================
def compute_ev_from_market_avg(df: pd.DataFrame, group_cols: list):
    """
    Adds:
      - Implied (per row)
      - MarketProb (mean implied across books for the selection)  ✅ uses transform (no merge)
      - ModelProb (defaults to MarketProb)
      - EV = (ModelProb - Implied)*100
    """
    if df.empty:
        return df

    df = df.copy()

    # Ensure numeric
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    # Normalize group cols to avoid dtype issues (Line often causes problems)
    for c in group_cols:
        if c not in df.columns:
            df[c] = np.nan
        # Use string representation for stable grouping keys (prevents merge dtype mismatch entirely)
        df[c] = df[c].astype("string")

    df["Implied"] = df["Price"].apply(american_to_implied)
    df["Implied"] = pd.to_numeric(df["Implied"], errors="coerce").fillna(0.5)
    df["Implied"] = clamp01(df["Implied"])

    # ✅ NO MERGE: transform avoids ValueError from dtype mismatch
    df["MarketProb"] = df.groupby(group_cols, dropna=False)["Implied"].transform("mean")
    df["MarketProb"] = pd.to_numeric(df["MarketProb"], errors="coerce").fillna(df["Implied"])
    df["MarketProb"] = clamp01(df["MarketProb"])

    # Baseline model prob: market avg (you can replace later with projections)
    df["ModelProb"] = df["MarketProb"]

    df["EV"] = (df["ModelProb"] - df["Implied"]) * 100.0
    return df

def best_price_group(df: pd.DataFrame, group_cols: list, odds_col="Price"):
    """
    Picks best American odds (max) within each selection group.
    """
    if df.empty:
        return df

    df = df.copy()
    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")
    df = df.dropna(subset=[odds_col])

    # group_cols are already string-normalized in compute_ev_from_market_avg
    idx = df.groupby(group_cols, dropna=False)[odds_col].idxmax()
    best = df.loc[idx].copy()

    best = best.rename(columns={"Book": "Best Book", odds_col: "Best Price"})
    return best

def ensure_probs_for_ev(df_best: pd.DataFrame):
    """
    Bulletproof: guarantees ModelProb/Implied exist.
    """
    if df_best.empty:
        return df_best

    df_best = df_best.copy()

    df_best["Best Price"] = pd.to_numeric(df_best["Best Price"], errors="coerce")
    df_best["Implied"] = df_best["Best Price"].apply(american_to_implied)
    df_best["Implied"] = pd.to_numeric(df_best["Implied"], errors="coerce").fillna(0.5)
    df_best["Implied"] = clamp01(df_best["Implied"])

    if "ModelProb" not in df_best.columns:
        df_best["ModelProb"] = df_best.get("MarketProb", df_best["Implied"])
    df_best["ModelProb"] = pd.to_numeric(df_best["ModelProb"], errors="coerce").fillna(df_best["Implied"])
    df_best["ModelProb"] = clamp01(df_best["ModelProb"])

    df_best["EV"] = (df_best["ModelProb"] - df_best["Implied"]) * 100.0
    return df_best

def enforce_no_contradictions(df_best: pd.DataFrame, key_cols: list):
    """
    Keeps only the top EV row per contradiction group to avoid both sides being recommended.
    """
    if df_best.empty:
        return df_best
    df = df_best.copy()
    df["EV"] = pd.to_numeric(df["EV"], errors="coerce").fillna(-1e9)
    keep_idx = df.groupby(key_cols, dropna=False)["EV"].idxmax()
    return df.loc[keep_idx].sort_values("EV", ascending=False)

def display_with_percents(df: pd.DataFrame, prob_cols=("ModelProb", "Implied", "MarketProb")):
    """
    Adds percent columns for display only (keeps original numeric for logic).
    """
    out = df.copy()
    for c in prob_cols:
        if c in out.columns:
            out[c + "_pct"] = pct(pd.to_numeric(out[c], errors="coerce").fillna(np.nan))
    return out

# =============================
# Odds API — Game lines (cached)
# =============================
@st.cache_data(ttl=60 * 60)  # 1 hour cache
def fetch_game_lines(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": ",".join(GAME_LINE_MARKETS.values()),
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    r = safe_get(url, params=params)
    payload = r.json() if "application/json" in (r.headers.get("Content-Type", "")) else r.text
    return r.status_code, payload

def normalize_game_lines(raw):
    rows = []
    if not is_list_of_dicts(raw):
        return pd.DataFrame(rows)

    for ev in raw:
        event = f"{ev.get('away_team')} @ {ev.get('home_team')}"
        commence = ev.get("commence_time")
        for bm in ev.get("bookmakers", []) or []:
            book = bm.get("title") or bm.get("key")
            for mk in bm.get("markets", []) or []:
                mkey = mk.get("key")
                for out in mk.get("outcomes", []) or []:
                    rows.append(
                        {
                            "Event": event,
                            "Commence": commence,
                            "Market": mkey,
                            "Side": out.get("name"),
                            "Line": out.get("point"),
                            "Price": out.get("price"),
                            "Book": book,
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["Market", "Side", "Price"])
    return df

# =============================
# Odds API — Player props (2-step: events -> event odds)
# =============================
@st.cache_data(ttl=60 * 60)  # 1 hour cache
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    r = safe_get(url, params=params)
    payload = r.json() if "application/json" in (r.headers.get("Content-Type", "")) else r.text
    return r.status_code, payload

@st.cache_data(ttl=60 * 60)  # 1 hour cache
def fetch_event_props(sport_key: str, event_id: str, market_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    r = safe_get(url, params=params)
    payload = r.json() if "application/json" in (r.headers.get("Content-Type", "")) else r.text
    return r.status_code, payload

def normalize_player_props(event_payload):
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame(rows)

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    event = f"{away} @ {home}"

    for bm in event_payload.get("bookmakers", []) or []:
        book = bm.get("title") or bm.get("key")
        for mk in bm.get("markets", []) or []:
            mkey = mk.get("key")
            for out in mk.get("outcomes", []) or []:
                # Depending on market, the Odds API outcome fields vary
                player = out.get("name")
                side = out.get("description") or out.get("name")  # often "Over/Under" in description
                line = out.get("point")
                price = out.get("price")

                rows.append(
                    {
                        "Event": event,
                        "Market": mkey,
                        "Player": player,
                        "Side": side,
                        "Line": line,
                        "Price": price,
                        "Book": book,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["Market", "Player", "Price"])
    return df

# =============================
# DataGolf — PGA (Win + Top10)
# =============================
@st.cache_data(ttl=60 * 60)
def datagolf_pre_tournament(tour="pga"):
    url = f"{DG_BASE}/preds/pre-tournament"
    params = {
        "tour": tour,
        "add_position": "10",
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_KEY,
    }
    r = safe_get(url, params=params)
    payload = r.json() if "application/json" in (r.headers.get("Content-Type", "")) else r.text
    return r.status_code, payload

@st.cache_data(ttl=60 * 60)
def datagolf_skill_ratings():
    url = f"{DG_BASE}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_KEY}
    r = safe_get(url, params=params)
    payload = r.json() if "application/json" in (r.headers.get("Content-Type", "")) else r.text
    return r.status_code, payload

def normalize_datagolf(pred_payload, skill_payload=None):
    if not isinstance(pred_payload, dict):
        return pd.DataFrame()

    list_keys = [k for k, v in pred_payload.items() if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict))]
    if not list_keys:
        return pd.DataFrame()

    main_key = sorted(list_keys, key=lambda k: len(pred_payload.get(k, [])), reverse=True)[0]
    df = pd.DataFrame(pred_payload.get(main_key, []))
    if df.empty:
        return df

    name_col = "player_name" if "player_name" in df.columns else ("player" if "player" in df.columns else None)
    if name_col:
        df = df.rename(columns={name_col: "Player"})
    else:
        df["Player"] = "Unknown"

    win_col = None
    top10_col = None
    for c in df.columns:
        lc = c.lower()
        if win_col is None and "win" in lc:
            win_col = c
        if top10_col is None and ("top_10" in lc or "top10" in lc):
            top10_col = c

    df["Win%"] = pd.to_numeric(df[win_col], errors="coerce") if win_col else np.nan
    df["Top10%"] = pd.to_numeric(df[top10_col], errors="coerce") if top10_col else np.nan

    if isinstance(skill_payload, dict):
        sk_list = None
        for _, v in skill_payload.items():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                sk_list = v
                break
        if sk_list:
            sk = pd.DataFrame(sk_list)
            sk_name = "player_name" if "player_name" in sk.columns else ("player" if "player" in sk.columns else None)
            if sk_name:
                sk = sk.rename(columns={sk_name: "Player"})
                cols = {c.lower(): c for c in sk.columns}

                putt_col = cols.get("sg_putt") or cols.get("sg_putting") or cols.get("putting")
                t2g_col = cols.get("sg_t2g") or cols.get("t2g")

                sk["SG_Putt"] = pd.to_numeric(sk[putt_col], errors="coerce") if putt_col else np.nan
                sk["SG_T2G"] = pd.to_numeric(sk[t2g_col], errors="coerce") if t2g_col else np.nan

                df = df.merge(sk[["Player", "SG_T2G", "SG_Putt"]].drop_duplicates("Player"), on="Player", how="left")

    df["SG_T2G_z"] = (df["SG_T2G"] - df["SG_T2G"].mean()) / (df["SG_T2G"].std() + 1e-9)
    df["SG_Putt_z"] = (df["SG_Putt"] - df["SG_Putt"].mean()) / (df["SG_Putt"].std() + 1e-9)

    df["Score_Win"] = df["Win%"].fillna(0) + df["SG_T2G_z"].fillna(0) * 0.6 + df["SG_Putt_z"].fillna(0) * 0.35
    df["Score_Top10"] = df["Top10%"].fillna(0) + df["SG_T2G_z"].fillna(0) * 0.45 + df["SG_Putt_z"].fillna(0) * 0.25

    keep = ["Player", "Win%", "Top10%", "SG_T2G", "SG_Putt", "Score_Win", "Score_Top10"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

# =============================
# Sidebar (clean)
# =============================
st.sidebar.markdown('<div class="big-title">Dashboard</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="subtle">Game lines • Player props • PGA</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

section = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA"], index=0)

sport = None
if section in ["Game Lines", "Player Props"]:
    sport = st.sidebar.selectbox("Sport", list(SPORT_KEYS.keys()), index=0)

debug = st.sidebar.checkbox("Show debug logs", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<span class="pill">Books: DK + FD</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="pill">Updated: {now_str()}</span>', unsafe_allow_html=True)

# =============================
# Header
# =============================
st.markdown('<div class="big-title">EdgeLedger</div>', unsafe_allow_html=True)
st.caption("Best Price • EV ranking • No-contradiction picks • Separate calls for game lines vs props")

# =============================
# MAIN
# =============================
if section == "Game Lines":
    sport_key = SPORT_KEYS[sport]
    market_label = st.selectbox("Bet Type", list(GAME_LINE_MARKETS.keys()), index=1)
    market_key = GAME_LINE_MARKETS[market_label]

    status, raw = fetch_game_lines(sport_key)
    debug_box(debug, f"Odds API /odds status={status}", raw)

    if status != 200 or not is_list_of_dicts(raw):
        st.error("No game line data returned from the API. Check key/limits/sport.")
        st.stop()

    df = normalize_game_lines(raw)
    if df.empty:
        st.warning("No game line rows were normalized from the API response.")
        st.stop()

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        st.warning("No rows for this market right now.")
        st.stop()

    group_cols = ["Event", "Market", "Side"]
    if market_key in ["totals", "spreads"]:
        group_cols += ["Line"]

    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")
    df_best = ensure_probs_for_ev(df_best)

    # No-contradiction rules
    if market_key == "totals":
        df_best = enforce_no_contradictions(df_best, ["Event", "Market", "Line"])
    else:
        df_best = enforce_no_contradictions(df_best, ["Event", "Market"])

    # Auto top 2–5
    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.subheader(f"{sport} — {market_label}")
    st.markdown("### Top Bets Ranked by EV")

    top_disp = display_with_percents(top_bets)
    cols = ["Event", "Side"] + (["Line"] if "Line" in top_disp.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb_pct", "Implied_pct", "EV"]
    cols = [c for c in cols if c in top_disp.columns]
    st.dataframe(top_disp[cols], use_container_width=True)

    st.markdown("### EV Bar Chart (Top Bets)")
    chart_df = top_disp[["Event", "EV"]].copy()
    chart_df = chart_df.set_index("Event")
    st.bar_chart(chart_df)

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)
    snap_disp = display_with_percents(snap)
    cols2 = ["Event", "Side"] + (["Line"] if "Line" in snap_disp.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb_pct", "Implied_pct", "EV"]
    cols2 = [c for c in cols2 if c in snap_disp.columns]
    st.dataframe(snap_disp[cols2], use_container_width=True)

elif section == "Player Props":
    sport_key = SPORT_KEYS[sport]
    prop_label = st.selectbox("Prop Type", list(PLAYER_PROP_MARKETS.keys()), index=0)
    prop_market = PLAYER_PROP_MARKETS[prop_label]

    e_status, events_raw = fetch_events(sport_key)
    debug_box(debug, f"Odds API /events status={e_status}", events_raw)

    if e_status != 200 or not is_list_of_dicts(events_raw):
        st.error("No events returned from the API. Check key/limits/sport.")
        st.stop()

    event_ids = [ev.get("id") for ev in events_raw if ev.get("id")]
    max_events_to_scan = st.sidebar.slider("Events to scan (usage control)", 1, 10, 5)
    event_ids = event_ids[:max_events_to_scan]

    if not event_ids:
        st.warning("No upcoming event IDs found.")
        st.stop()

    all_props = []
    debug_rows = []

    for eid in event_ids:
        p_status, payload = fetch_event_props(sport_key, eid, prop_market)
        debug_rows.append({"event_id": eid, "market": prop_market, "status": p_status, "ok": (p_status == 200)})

        if p_status != 200 or not isinstance(payload, dict):
            continue

        dfp = normalize_player_props(payload)
        if not dfp.empty:
            all_props.append(dfp)

        time.sleep(0.12)

    debug_box(debug, "Props per-event call results", debug_rows)

    if not all_props:
        st.warning(
            "No player props returned from DraftKings/FanDuel for the scanned events.\n\n"
            "Try a different prop type (or scan more events)."
        )
        st.stop()

    df = pd.concat(all_props, ignore_index=True)

    # Keep player-based outcomes; drop obvious junk
    df["Player"] = df["Player"].astype(str)
    df["Side"] = df["Side"].astype(str)
    df = df.dropna(subset=["Price"])
    df = df[df["Player"].str.len() > 1]

    # Grouping for prop selections
    group_cols = ["Event", "Market", "Player", "Side"]
    if "Line" in df.columns:
        group_cols += ["Line"]

    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")
    df_best = ensure_probs_for_ev(df_best)

    # ✅ No-contradictions (keep only best EV per player+line)
    contradiction_cols = ["Event", "Market", "Player"] + (["Line"] if "Line" in df_best.columns else [])
    df_best = enforce_no_contradictions(df_best, contradiction_cols)

    # Auto top 2–5
    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.subheader(f"{sport} — Player Props")
    st.caption(f"Prop Type: {prop_label} • Scanned events: {len(event_ids)} • Normalized props: {len(df_best)}")

    st.markdown("### Top Bets Ranked by EV (No Contradictions)")
    top_disp = display_with_percents(top_bets)
    cols = ["Event", "Player", "Side"] + (["Line"] if "Line" in top_disp.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb_pct", "Implied_pct", "EV"]
    cols = [c for c in cols if c in top_disp.columns]
    st.dataframe(top_disp[cols], use_container_width=True)

    st.markdown("### Model vs Implied (Top Bets) — Percent")
    # chart: percent columns (display only)
    chart_df = top_disp[["Player", "ModelProb_pct", "Implied_pct"]].copy()
    chart_df = chart_df.set_index("Player")
    st.bar_chart(chart_df)

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)
    snap_disp = display_with_percents(snap)
    cols2 = ["Event", "Player", "Side"] + (["Line"] if "Line" in snap_disp.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb_pct", "Implied_pct", "EV"]
    cols2 = [c for c in cols2 if c in snap_disp.columns]
    st.dataframe(snap_disp[cols2], use_container_width=True)

else:
    st.subheader("PGA — Picks (Win + Top-10)")
    st.caption("Uses DataGolf pre-tournament probabilities + skill ratings (tee-to-green + putting) when available.")

    p_status, preds = datagolf_pre_tournament(tour="pga")
    s_status, skills = datagolf_skill_ratings()

    debug_box(debug, f"DataGolf pre-tournament status={p_status}", preds)
    debug_box(debug, f"DataGolf skill-ratings status={s_status}", skills)

    if p_status != 200 or not isinstance(preds, dict):
        st.error("No DataGolf predictions available (check key/access).")
        st.stop()

    skill_payload = skills if (s_status == 200 and isinstance(skills, dict)) else None
    df = normalize_datagolf(preds, skill_payload=skill_payload)

    if df.empty:
        st.warning("No PGA rows could be normalized from the DataGolf response.")
        st.stop()

    win_top5 = df.sort_values("Score_Win", ascending=False).head(5).copy()
    top10_top5 = df.sort_values("Score_Top10", ascending=False).head(5).copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🏆 Top 5 — Win")
        st.dataframe(win_top5, use_container_width=True)

    with c2:
        st.markdown("### 🎯 Top 5 — Top-10")
        st.dataframe(top10_top5, use_container_width=True)

    st.markdown("### Snapshot — Top 25 (Win Score)")
    st.dataframe(df.sort_values("Score_Win", ascending=False).head(25), use_container_width=True)
