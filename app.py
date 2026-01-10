# app.py — EdgeLedger Dashboard (NFL / CFB / CBB)
# ✅ The Odds API (your key) with production-friendly controls
# ✅ Dropdowns ALWAYS switch datasets immediately (per-selection snapshots)
# ✅ API calls ONLY happen when you click "Load / Refresh Data" (credit friendly)
# ✅ Separate API flows:
#    - Game Lines: /sports/{sport_key}/odds?markets=h2h,spreads,totals
#    - Player Props: /sports/{sport_key}/events then /events/{event_id}/odds (ONE market per request)
# ✅ Snapshot fallback keeps site operational when quota is hit
# ✅ Best price across books + EV + Top 2–5 ranked bets
# ✅ Dynamic prop market discovery per sport (cached + snapshot)
# ✅ Robust diagnostics + no "stuck" dropdown behavior
#
# NOTE: This hard-codes your key as requested. For public apps, move it to Streamlit Secrets.

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# PAGE / THEME
# -----------------------------
st.set_page_config(page_title="EdgeLedger Dashboard", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
section[data-testid="stSidebar"] {background: #0b1220;}
section[data-testid="stSidebar"] * {color: #e8eefc;}
h1,h2,h3 {letter-spacing:.2px;}
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px; border-radius: 14px;
}
.small-note {opacity: .75; font-size: 0.92rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎯 EdgeLedger Dashboard")

# -----------------------------
# API CONFIG (YOUR KEY)
# -----------------------------
API_KEY = "6a5d08e7c2407da6fb95b86ad9619bf0"
BASE_URL = "https://api.the-odds-api.com/v4"
REGION = "us"
ODDS_FORMAT = "american"
HEADERS = {"User-Agent": "EdgeLedgerDashboard/4.0"}

SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}

GAME_MARKETS = {
    "Moneyline (H2H)": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

# Some accounts return better bookmaker coverage when filtered.
FALLBACK_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "pinnacle", "circa"]

# -----------------------------
# CREDIT-FRIENDLY SETTINGS
# -----------------------------
REQUIRE_CLICK_TO_LOAD = True  # only fetch from API when user clicks

GAME_LINES_TTL = 15 * 60      # 15 min cache
EVENTS_TTL = 30 * 60          # 30 min cache
PROPS_TTL = 15 * 60           # 15 min cache
MARKETS_TTL = 24 * 60 * 60    # 24 hrs cache (markets rarely change)

# Snapshot folder (persisted during runtime; may reset on redeploy)
SNAP_DIR = ".snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# -----------------------------
# HELPERS
# -----------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def snap_path(name: str) -> str:
    return os.path.join(SNAP_DIR, name)


def save_snapshot_json(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_snapshot_json(path: str) -> Any:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_snapshot_df(path: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Parquet not always available — fallback to CSV
        try:
            df.to_csv(path.replace(".parquet", ".csv"), index=False)
        except Exception:
            pass


def load_snapshot_df(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
        csv_path = path.replace(".parquet", ".csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def american_to_implied(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def consensus_true_prob(df_group: pd.DataFrame) -> float:
    vals = df_group["Implied"].dropna().values
    if len(vals) == 0:
        return np.nan
    return float(np.clip(vals.mean(), 0.02, 0.98))


def pretty_market_name(k: str) -> str:
    return (k or "").replace("_", " ").title()


def best_price_table(df: pd.DataFrame, group_cols: List[str], price_col: str = "Price") -> pd.DataFrame:
    """Best price across books for American odds = numeric max."""
    if df.empty:
        return df
    idx = df.groupby(group_cols)[price_col].idxmax()
    best = df.loc[idx].copy()
    best = best.rename(columns={price_col: "Best Price", "Book": "Best Book"})
    return best.reset_index(drop=True)


def is_out_of_credits(status: int, payload: Any) -> bool:
    if status != 401 or not isinstance(payload, dict):
        return False
    return payload.get("error_code") in ("OUT_OF_USAGE_CREDITS", "OUT_OF_CREDITS")


def safe_get(url: str, params: dict, timeout: int = 20) -> Tuple[bool, Any, int]:
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        status = r.status_code
        try:
            payload = r.json()
        except Exception:
            payload = {"message": r.text}

        if status == 200:
            return True, payload, status
        return False, payload, status
    except Exception as e:
        return False, {"message": str(e)}, -1


# -----------------------------
# NORMALIZERS
# -----------------------------
def normalize_game_lines(raw: list) -> pd.DataFrame:
    rows = []
    if not isinstance(raw, list):
        return pd.DataFrame()

    for ev in raw:
        if not isinstance(ev, dict):
            continue

        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        event_name = f"{away} @ {home}".strip(" @")
        commence = ev.get("commence_time")

        for bk in ev.get("bookmakers", []) or []:
            book = bk.get("title") or bk.get("key")
            for mk in bk.get("markets", []) or []:
                mkey = mk.get("key")
                for out in mk.get("outcomes", []) or []:
                    rows.append(
                        {
                            "Event": event_name,
                            "Commence": commence,
                            "Market": mkey,
                            "Outcome": out.get("name"),
                            "Line": out.get("point"),
                            "Price": out.get("price"),
                            "Book": book,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)
    return df


def normalize_event_props(event_odds_json: dict, market_key: str) -> pd.DataFrame:
    if not isinstance(event_odds_json, dict):
        return pd.DataFrame()

    home = event_odds_json.get("home_team", "")
    away = event_odds_json.get("away_team", "")
    event_name = f"{away} @ {home}".strip(" @")
    commence = event_odds_json.get("commence_time")

    rows = []
    for bk in event_odds_json.get("bookmakers", []) or []:
        book = bk.get("title") or bk.get("key")
        for mk in bk.get("markets", []) or []:
            if mk.get("key") != market_key:
                continue
            for out in mk.get("outcomes", []) or []:
                player = (
                    out.get("description")
                    or out.get("participant")
                    or out.get("player")
                    or ""
                )
                player = (player or "").strip()
                if not player:
                    continue
                rows.append(
                    {
                        "Event": event_name,
                        "Commence": commence,
                        "Market": market_key,
                        "Player": player,
                        "Side": (out.get("name") or "").strip(),
                        "Line": out.get("point"),
                        "Price": out.get("price"),
                        "Book": book,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)
    df["Player"] = df["Player"].fillna("").astype(str).str.strip()
    df["Side"] = df["Side"].fillna("").astype(str)
    return df


# -----------------------------
# API CALLS (SEPARATED + CACHED)
# -----------------------------
@st.cache_data(ttl=MARKETS_TTL, show_spinner=False)
def api_discover_markets(sport_key: str) -> Tuple[List[dict], dict]:
    url = f"{BASE_URL}/sports/{sport_key}/markets"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    dbg = {
        "endpoint": "markets",
        "url": url,
        "status": status,
        "ok": ok,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
        "error": None if ok else payload,
    }
    return (payload if ok and isinstance(payload, list) else []), dbg


@st.cache_data(ttl=GAME_LINES_TTL, show_spinner=False)
def api_fetch_game_lines_once(sport_key: str, markets: List[str], bookmakers: List[str] | None = None):
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)

    ok, payload, status = safe_get(url, params=params)
    dbg = {
        "endpoint": "odds(game_lines)",
        "url": url,
        "status": status,
        "ok": ok,
        "markets": markets,
        "bookmakers_filter": bookmakers,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
        "error": None if ok else payload,
        "rows_normalized": 0,
    }

    if ok and isinstance(payload, list):
        df = normalize_game_lines(payload)
        dbg["rows_normalized"] = int(df.shape[0])
        return df, dbg

    return pd.DataFrame(), dbg


def api_fetch_game_lines_with_fallback(sport_key: str, markets: List[str]):
    df1, dbg1 = api_fetch_game_lines_once(sport_key, markets, bookmakers=None)
    if not df1.empty:
        dbg1["fallback_used"] = False
        return df1, dbg1
    df2, dbg2 = api_fetch_game_lines_once(sport_key, markets, bookmakers=FALLBACK_BOOKMAKERS)
    dbg2["fallback_used"] = True
    dbg2["first_attempt"] = dbg1
    return df2, dbg2


@st.cache_data(ttl=EVENTS_TTL, show_spinner=False)
def api_fetch_events(sport_key: str) -> Tuple[List[dict], dict]:
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    dbg = {
        "endpoint": "events",
        "url": url,
        "status": status,
        "ok": ok,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
        "error": None if ok else payload,
    }
    return (payload if ok and isinstance(payload, list) else []), dbg


@st.cache_data(ttl=PROPS_TTL, show_spinner=False)
def api_fetch_event_props(sport_key: str, event_id: str, market_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": market_key,  # ONE market per request (prevents 422)
        "oddsFormat": ODDS_FORMAT,
    }
    ok, payload, status = safe_get(url, params=params)
    dbg = {
        "endpoint": "event_odds(player_props)",
        "url": url,
        "status": status,
        "ok": ok,
        "market": market_key,
        "payload_type": type(payload).__name__,
        "error": None if ok else payload,
        "rows_normalized": 0,
    }
    if ok and isinstance(payload, dict):
        df = normalize_event_props(payload, market_key)
        dbg["rows_normalized"] = int(df.shape[0])
        return df, dbg
    return pd.DataFrame(), dbg


# -----------------------------
# SIDEBAR (CLEANED)
# -----------------------------
st.sidebar.title("📊 Dashboard")

sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_name]

view = st.sidebar.radio("Bet Type", ["Game Lines", "Player Props"], index=0)

top_n = st.sidebar.slider("Top Bets (EV)", 2, 5, 3, 1)

debug_mode = st.sidebar.toggle("Show Debug", value=False)

st.sidebar.markdown("---")

load_clicked = False
if REQUIRE_CLICK_TO_LOAD:
    load_clicked = st.sidebar.button("📥 Load / Refresh Data")
else:
    load_clicked = True

if st.sidebar.button("🔄 Force refresh (clear cache)"):
    st.cache_data.clear()
    st.rerun()

# Header metrics
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Sport", sport_name)
c2.metric("Last Render", now_utc())
c3.metric("Mode", view)

tabs = st.tabs(["Dashboard", "All Books (Raw)", "Diagnostics / Help"])

# -----------------------------
# COMMON: selection rerun (UI fixes for dropdowns)
# -----------------------------
# We store the selection and rerun to ensure the correct per-selection snapshot loads immediately.
selection_key = {"sport_key": sport_key, "view": view, "top_n": int(top_n)}
if "last_selection_key" not in st.session_state:
    st.session_state["last_selection_key"] = None

# -----------------------------
# GAME LINES VIEW
# -----------------------------
if view == "Game Lines":
    chosen_human = st.sidebar.multiselect(
        "Markets",
        list(GAME_MARKETS.keys()),
        default=["Spreads", "Totals", "Moneyline (H2H)"],
    )
    chosen = [GAME_MARKETS[x] for x in chosen_human] if chosen_human else ["h2h", "spreads", "totals"]
    markets_sig = "_".join(sorted(chosen))

    # Update selection key so changing markets triggers correct snapshot immediately
    selection_key["markets_sig"] = markets_sig

    # If selection changed, rerun (NO API fetch unless clicked)
    if st.session_state["last_selection_key"] != selection_key:
        st.session_state["last_selection_key"] = selection_key
        st.rerun()

    snap_df_path = snap_path(f"lines_{sport_key}_{markets_sig}.parquet")
    snap_dbg_path = snap_path(f"lines_dbg_{sport_key}_{markets_sig}.json")

    # Always show snapshot if not clicking
    if REQUIRE_CLICK_TO_LOAD and not load_clicked:
        st.info("Viewing cached/snapshot data for this selection. Click **📥 Load / Refresh Data** for fresh odds.")
        df_lines = load_snapshot_df(snap_df_path)
        dbg_lines = load_snapshot_json(snap_dbg_path) or {"snapshot_used": True}

        if df_lines.empty:
            st.warning("No saved snapshot exists yet for these settings. Click **Load / Refresh Data**.")
            if debug_mode:
                with tabs[2]:
                    st.write(dbg_lines)
            st.stop()
    else:
        # Attempt API fetch, then snapshot, else fallback to snapshot
        df_lines, dbg_lines = api_fetch_game_lines_with_fallback(sport_key, chosen)

        if (not dbg_lines.get("ok", True)) and is_out_of_credits(dbg_lines.get("status", 0), dbg_lines.get("error")):
            st.warning("🚫 Odds API credits exhausted. Showing last snapshot if available.")
            df_lines = load_snapshot_df(snap_df_path)
            dbg_lines["snapshot_used"] = True

        if not df_lines.empty:
            save_snapshot_df(snap_df_path, df_lines)
            save_snapshot_json(snap_dbg_path, dbg_lines)

        if df_lines.empty:
            st.warning("No game lines available from API; showing last snapshot if available.")
            df_lines = load_snapshot_df(snap_df_path)

        if df_lines.empty:
            st.error("No game line data available right now (and no snapshot exists yet).")
            with tabs[2]:
                st.write(dbg_lines)
            st.stop()

    raw = df_lines.copy()

    best = best_price_table(raw, ["Event", "Market", "Outcome", "Line"], "Price")
    best["Best Implied"] = best["Best Price"].apply(american_to_implied)

    cons = (
        raw.groupby(["Event", "Market", "Outcome", "Line"])
        .apply(consensus_true_prob)
        .rename("P_true")
        .reset_index()
    )
    best = best.merge(cons, on=["Event", "Market", "Outcome", "Line"], how="left")
    best["EV"] = (best["P_true"] - best["Best Implied"]) * 100.0
    best = best.sort_values("EV", ascending=False)

    top = best.head(top_n).copy()
    top.insert(0, "⭐", "⭐")

    with tabs[0]:
        st.subheader("🔥 Auto-Ranked Top Bets (Game Lines)")
        st.caption("Best Book = absolute best bookmaker for that exact line/outcome. EV uses market consensus implied baseline.")
        st.dataframe(
            top[["⭐", "Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )
        st.subheader("Best Price Board (Game Lines)")
        st.dataframe(
            best[["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("All Books (Raw Game Lines)")
        st.dataframe(raw[["Event", "Market", "Outcome", "Line", "Price", "Book", "Implied"]], use_container_width=True)

    with tabs[2]:
        st.subheader("Diagnostics (Game Lines)")
        if debug_mode:
            st.write(load_snapshot_json(snap_dbg_path) or {})
        else:
            st.markdown('<div class="small-note">Enable "Show Debug" in the sidebar to see API diagnostics.</div>', unsafe_allow_html=True)

# -----------------------------
# PLAYER PROPS VIEW
# -----------------------------
else:
    # Discover prop markets (cached + snapshot)
    snap_markets_path = snap_path(f"markets_{sport_key}.json")

    markets_list: List[dict] = []
    dbg_mk: Dict[str, Any] = {}

    # For dropdown usability, we try to load markets snapshot immediately (no click needed)
    markets_list = load_snapshot_json(snap_markets_path) or []

    # If user clicked refresh, attempt fresh discovery
    if load_clicked or (not markets_list):
        markets_list, dbg_mk = api_discover_markets(sport_key)
        if markets_list:
            save_snapshot_json(snap_markets_path, markets_list)

    discovered = {}
    for m in markets_list:
        if isinstance(m, dict):
            k = m.get("key") or m.get("market_key")
            n = m.get("name") or m.get("title") or pretty_market_name(k)
            if k:
                discovered[k] = n

    player_markets = sorted([k for k in discovered.keys() if k.startswith("player_") and k != "player_props"])

    if not player_markets:
        st.warning("No player prop markets are available (or none saved yet).")
        with tabs[2]:
            st.write({"markets_dbg": dbg_mk, "saved_markets": bool(markets_list)})
        st.stop()

    prop_market_key = st.sidebar.selectbox(
        "Prop Market",
        options=player_markets,
        format_func=lambda k: discovered.get(k, pretty_market_name(k)),
        index=0,
    )

    # Update selection key so changing prop market triggers correct snapshot immediately
    selection_key["prop_market_key"] = prop_market_key

    # If selection changed, rerun (NO API fetch unless clicked)
    if st.session_state["last_selection_key"] != selection_key:
        st.session_state["last_selection_key"] = selection_key
        st.rerun()

    snap_best_path = snap_path(f"props_best_{sport_key}_{prop_market_key}.parquet")
    snap_raw_path = snap_path(f"props_raw_{sport_key}_{prop_market_key}.parquet")
    snap_dbg_path = snap_path(f"props_dbg_{sport_key}_{prop_market_key}.json")

    # If not clicked, show snapshot immediately for that selection
    if REQUIRE_CLICK_TO_LOAD and not load_clicked:
        st.info("Viewing cached/snapshot props for this selection. Click **📥 Load / Refresh Data** for fresh props.")
        best_snap = load_snapshot_df(snap_best_path)
        raw_snap = load_snapshot_df(snap_raw_path)
        dbg_snap = load_snapshot_json(snap_dbg_path) or {"snapshot_used": True}

        if best_snap.empty:
            st.warning("No saved snapshot exists yet for this prop market. Click **Load / Refresh Data**.")
            if debug_mode:
                with tabs[2]:
                    st.write(dbg_snap)
            st.stop()

        top = best_snap.sort_values("EV", ascending=False).head(top_n).copy()
        top.insert(0, "⭐", "⭐")

        with tabs[0]:
            st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {discovered.get(prop_market_key)}")
            st.dataframe(
                top[["⭐", "Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
                use_container_width=True,
            )
            st.subheader("Best Price Board (Player Props)")
            st.dataframe(
                best_snap[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
                use_container_width=True,
            )

        with tabs[1]:
            st.subheader("All Books (Raw Player Props)")
            st.dataframe(
                raw_snap[["Event", "Market", "Player", "Side", "Line", "Price", "Book", "Implied"]],
                use_container_width=True,
            )

        with tabs[2]:
            st.subheader("Diagnostics (Player Props)")
            if debug_mode:
                st.write(dbg_snap)
            else:
                st.markdown('<div class="small-note">Enable "Show Debug" in the sidebar to see diagnostics.</div>', unsafe_allow_html=True)

        st.stop()

    # Otherwise, fetch fresh via separate API flow: events -> event odds
    events, dbg_ev = api_fetch_events(sport_key)

    if not events:
        st.warning("No events returned for this sport right now.")
        with tabs[2]:
            if debug_mode:
                st.write(dbg_ev)
        # fallback to snapshot if exists
        best_snap = load_snapshot_df(snap_best_path)
        if not best_snap.empty:
            st.info("Showing last snapshot instead.")
            top = best_snap.sort_values("EV", ascending=False).head(top_n).copy()
            top.insert(0, "⭐", "⭐")
            with tabs[0]:
                st.dataframe(top, use_container_width=True)
        st.stop()

    all_props = []
    prop_diag = {"attempted_events": 0, "events_with_rows": 0, "skipped_empty": 0, "errors": 0, "stopped_due_to_credits": False}

    with st.spinner("Loading player props (event-by-event)…"):
        max_events_to_try = min(20, len(events))  # protects credits
        for e in events[:max_events_to_try]:
            event_id = e.get("id")
            if not event_id:
                prop_diag["skipped_empty"] += 1
                continue
            prop_diag["attempted_events"] += 1

            df_evprops, dbg_p = api_fetch_event_props(sport_key, event_id, prop_market_key)

            if not dbg_p.get("ok", True):
                prop_diag["errors"] += 1
                if is_out_of_credits(dbg_p.get("status", 0), dbg_p.get("error")):
                    prop_diag["stopped_due_to_credits"] = True
                    break

            if df_evprops.empty:
                prop_diag["skipped_empty"] += 1
                continue

            prop_diag["events_with_rows"] += 1
            all_props.append(df_evprops)

    if not all_props:
        st.warning("No props rows were normalized for that market right now.")
        st.info("Try another market in the dropdown — availability varies by sport/date/book.")
        with tabs[2]:
            if debug_mode:
                st.write({"events_dbg": dbg_ev, "prop_diag": prop_diag})
        # show snapshot fallback if exists
        best_snap = load_snapshot_df(snap_best_path)
        if not best_snap.empty:
            st.info("Showing last snapshot instead.")
            top = best_snap.sort_values("EV", ascending=False).head(top_n).copy()
            top.insert(0, "⭐", "⭐")
            with tabs[0]:
                st.dataframe(top, use_container_width=True)
        st.stop()

    raw_props = pd.concat(all_props, ignore_index=True)
    raw_props["Player"] = raw_props["Player"].fillna("").astype(str).str.strip()
    raw_props = raw_props[raw_props["Player"].str.len() > 0].copy()

    group_cols = ["Event", "Market", "Player", "Side", "Line"]

    cons = raw_props.groupby(group_cols).apply(consensus_true_prob).rename("P_true").reset_index()
    best = best_price_table(raw_props, group_cols, "Price")
    best["Best Implied"] = best["Best Price"].apply(american_to_implied)
    best = best.merge(cons, on=group_cols, how="left")
    best["EV"] = (best["P_true"] - best["Best Implied"]) * 100.0
    best = best.sort_values("EV", ascending=False)

    # Save per-selection snapshots
    save_snapshot_df(snap_best_path, best)
    save_snapshot_df(snap_raw_path, raw_props)
    save_snapshot_json(snap_dbg_path, {"events_dbg": dbg_ev, "prop_diag": prop_diag, "market": prop_market_key})

    top = best.head(top_n).copy()
    top.insert(0, "⭐", "⭐")

    with tabs[0]:
        st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {discovered.get(prop_market_key)}")
        st.caption("Best Book = absolute best bookmaker per player/side/line. EV uses market consensus implied baseline.")
        st.dataframe(
            top[["⭐", "Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )

        st.subheader("Best Price Board (Player Props)")
        st.dataframe(
            best[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )

        with st.expander("Prop Diagnostics"):
            st.write(prop_diag)

    with tabs[1]:
        st.subheader("All Books (Raw Player Props)")
        st.dataframe(
            raw_props[["Event", "Market", "Player", "Side", "Line", "Price", "Book", "Implied"]],
            use_container_width=True,
        )

    with tabs[2]:
        st.subheader("Diagnostics (Player Props)")
        if debug_mode:
            st.write({"market_dbg": dbg_mk, "events_dbg": dbg_ev, "prop_diag": prop_diag})
        else:
            st.markdown('<div class="small-note">Enable "Show Debug" in the sidebar to see diagnostics.</div>', unsafe_allow_html=True)

# -----------------------------
# HELP / DEPLOY
# -----------------------------
with tabs[2]:
    st.subheader("Deploy Notes (Operational Site)")
    st.markdown(
        """
### Streamlit Cloud deploy
1) Push `app.py` to GitHub  
2) Streamlit Cloud → **New app** → select repo → main file: `app.py`  
3) Deploy

### Why dropdowns now work
- Snapshots are saved **per selection** (sport + view + market settings)
- Changing a dropdown triggers a rerun to load the correct snapshot instantly
- API fetch only occurs when you click **📥 Load / Refresh Data**

### If you hit OUT_OF_USAGE_CREDITS
Upgrade your plan or wait for quota reset, then click **Load / Refresh Data**.
"""
    )
