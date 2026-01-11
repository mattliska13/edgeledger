import os
import time
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import streamlit as st

# =============================
# Page + Theme polish
# =============================
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; }
      [data-testid="stSidebarNav"] { display: none; } /* hides multipage nav */
      .big-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; }
      .subtle { color: rgba(250,250,250,0.75); }
      .pill { display:inline-block; padding:0.2rem 0.6rem; border-radius:999px; background:rgba(255,255,255,0.08); margin-right:0.4rem; }
      .metric-card { padding: 0.75rem 1rem; border-radius: 16px; background: rgba(255,255,255,0.06); }
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

PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_passing_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rushing_yds",
    "Receiving Yards": "player_receiving_yds",
    "Receptions": "player_receptions",
}

DG_BASE = "https://feeds.datagolf.com"

# =============================
# Utility + Debug
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
        # Fake a response-like object for safe downstream handling
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

# =============================
# EV + Best price (FIXED / BULLETPROOF)
# =============================
def compute_ev_from_market_avg(df: pd.DataFrame, group_cols: list):
    """
    Adds: Implied, MarketProb, ModelProb, EV
    ModelProb defaults to MarketProb (market-average implied).
    """
    if df.empty:
        return df

    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])

    df["Implied"] = df["Price"].apply(american_to_implied)

    avg = (
        df.groupby(group_cols, dropna=False)["Implied"]
        .mean()
        .reset_index()
        .rename(columns={"Implied": "MarketProb"})
    )
    df = df.merge(avg, on=group_cols, how="left")

    df["ModelProb"] = df["MarketProb"]
    df["ModelProb"] = pd.to_numeric(df["ModelProb"], errors="coerce").clip(0.001, 0.999)
    df["Implied"] = pd.to_numeric(df["Implied"], errors="coerce").clip(0.001, 0.999)

    df["EV"] = (df["ModelProb"] - df["Implied"]) * 100.0
    return df

def best_price_group(df: pd.DataFrame, group_cols: list, odds_col="Price"):
    """
    Picks the best American odds (max) within each selection group.
    Keeps MarketProb/ModelProb if they exist (because we select from df rows).
    """
    if df.empty:
        return df

    df = df.copy()
    df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")
    df = df.dropna(subset=[odds_col])

    idx = df.groupby(group_cols, dropna=False)[odds_col].idxmax()
    best = df.loc[idx].copy()

    best = best.rename(columns={"Book": "Best Book", odds_col: "Best Price"})
    return best

def ensure_probs_for_ev(df_best: pd.DataFrame, group_cols: list, df_full: pd.DataFrame | None = None):
    """
    Ensures df_best has MarketProb and ModelProb; prevents KeyError forever.
    """
    if df_best.empty:
        return df_best

    df_best = df_best.copy()

    # If we can merge missing columns from df_full (recommended)
    if df_full is not None and not df_full.empty:
        need = []
        for c in ["MarketProb", "ModelProb"]:
            if c not in df_best.columns and c in df_full.columns:
                need.append(c)
        if need:
            src = df_full[group_cols + need].drop_duplicates(group_cols)
            df_best = df_best.merge(src, on=group_cols, how="left")

    # Ensure Implied exists from Best Price
    df_best["Implied"] = pd.to_numeric(df_best.get("Best Price"), errors="coerce").apply(american_to_implied)

    if "MarketProb" not in df_best.columns:
        df_best["MarketProb"] = df_best["Implied"]

    if "ModelProb" not in df_best.columns:
        df_best["ModelProb"] = df_best["MarketProb"]

    df_best["ModelProb"] = pd.to_numeric(df_best["ModelProb"], errors="coerce").fillna(df_best["Implied"]).clip(0.001, 0.999)
    df_best["Implied"] = pd.to_numeric(df_best["Implied"], errors="coerce").fillna(0.5).clip(0.001, 0.999)

    df_best["EV"] = (df_best["ModelProb"] - df_best["Implied"]) * 100.0
    return df_best

def enforce_no_contradictions(df_best: pd.DataFrame, contradiction_key_cols: list):
    """
    Keeps ONLY the best EV per contradiction group so we never recommend both sides.
    """
    if df_best.empty:
        return df_best

    df = df_best.copy()
    df["EV"] = pd.to_numeric(df["EV"], errors="coerce").fillna(-1e9)
    keep_idx = df.groupby(contradiction_key_cols, dropna=False)["EV"].idxmax()
    return df.loc[keep_idx].sort_values("EV", ascending=False)

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
# Odds API — Player props (2-step events -> event odds)
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
                player = out.get("name")
                side = out.get("description") or out.get("name")
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

    # Find biggest list-like table
    list_keys = [k for k, v in pred_payload.items() if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict))]
    if not list_keys:
        return pd.DataFrame()

    main_key = sorted(list_keys, key=lambda k: len(pred_payload.get(k, [])), reverse=True)[0]
    df = pd.DataFrame(pred_payload.get(main_key, []))
    if df.empty:
        return df

    # Player name
    name_col = "player_name" if "player_name" in df.columns else ("player" if "player" in df.columns else None)
    if name_col:
        df = df.rename(columns={name_col: "Player"})
    else:
        df["Player"] = "Unknown"

    # Prob columns heuristics
    win_col = None
    top10_col = None
    for c in df.columns:
        lc = c.lower()
        if ("course" in lc or "fit" in lc) and "win" in lc:
            win_col = c
        if ("course" in lc or "fit" in lc) and ("top_10" in lc or "top10" in lc):
            top10_col = c

    if not win_col:
        for c in df.columns:
            if "win" in c.lower():
                win_col = c
                break

    if not top10_col:
        for c in df.columns:
            if "top_10" in c.lower() or "top10" in c.lower():
                top10_col = c
                break

    df["Win%"] = pd.to_numeric(df[win_col], errors="coerce") if win_col else np.nan
    df["Top10%"] = pd.to_numeric(df[top10_col], errors="coerce") if top10_col else np.nan

    # Skill merge (SG T2G + Putting where possible)
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

                ott_col = cols.get("sg_ott") or cols.get("ott") or cols.get("off_the_tee")
                app_col = cols.get("sg_app") or cols.get("approach") or cols.get("app")
                arg_col = cols.get("sg_arg") or cols.get("around_green") or cols.get("arg")

                sk["SG_Putt"] = pd.to_numeric(sk[putt_col], errors="coerce") if putt_col else np.nan
                if t2g_col:
                    sk["SG_T2G"] = pd.to_numeric(sk[t2g_col], errors="coerce")
                else:
                    parts = []
                    for cc in [ott_col, app_col, arg_col]:
                        if cc:
                            parts.append(pd.to_numeric(sk[cc], errors="coerce"))
                    sk["SG_T2G"] = np.nan if not parts else np.nanmean(np.vstack([p.to_numpy() for p in parts]), axis=0)

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

    group_cols = ["Event", "Market", "Side"] + (["Line"] if market_key in ["totals", "spreads"] else [])

    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")

    # ✅ FIX: ensures ModelProb exists (prevents KeyError)
    df_best = ensure_probs_for_ev(df_best, group_cols=group_cols, df_full=df_ev)

    # No-contradiction
    if market_key == "totals":
        df_best = enforce_no_contradictions(df_best, ["Event", "Market", "Line"])
    else:
        df_best = enforce_no_contradictions(df_best, ["Event", "Market"])

    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.subheader(f"{sport} — {market_label}")
    st.markdown("### Top Bets Ranked by EV")

    cols = ["Event", "Side"] + (["Line"] if "Line" in top_bets.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(top_bets[cols], use_container_width=True)

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)
    cols2 = ["Event", "Side"] + (["Line"] if "Line" in snap.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(snap[cols2], use_container_width=True)

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
            "No player props were returned for this prop type from DraftKings/FanDuel for the scanned events.\n\n"
            "This usually means the market isn't posted yet for those events/books."
        )
        st.stop()

    df = pd.concat(all_props, ignore_index=True)

    # Ensure player-based outcomes
    df["Player"] = df["Player"].astype(str)
    df = df[~df["Player"].str.lower().isin(["over", "under"])].dropna(subset=["Price"])

    group_cols = ["Event", "Market", "Player", "Side"] + (["Line"] if "Line" in df.columns else [])

    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")

    # ✅ FIX: ensures ModelProb exists (prevents KeyError)
    df_best = ensure_probs_for_ev(df_best, group_cols=group_cols, df_full=df_ev)

    # No-contradictions: keep only best EV per player+line+market
    contradiction_cols = ["Event", "Market", "Player"] + (["Line"] if "Line" in df_best.columns else [])
    df_best = enforce_no_contradictions(df_best, contradiction_cols)

    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top).copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.subheader(f"{sport} — Player Props")
    st.caption(f"Prop Type: {prop_label} • Scanned events: {len(event_ids)} • Normalized props: {len(df_best)}")

    st.markdown("### Top Bets Ranked by EV (No Contradictions)")
    cols = ["Event", "Player", "Side"] + (["Line"] if "Line" in top_bets.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(top_bets[cols], use_container_width=True)

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)
    cols2 = ["Event", "Player", "Side"] + (["Line"] if "Line" in snap.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(snap[cols2], use_container_width=True)

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
