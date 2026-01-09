# app.py — EdgeLedger Dashboard (NFL / CFB / CBB)
# ✅ The Odds API, optimized for free/limited credits:
#   - Click-to-load (no automatic API calls on reruns)
#   - Aggressive caching
#   - Snapshot fallback (keeps site operational when quota is exhausted)
# ✅ Separate API calls:
#   - Game Lines: /sports/{sport_key}/odds?markets=h2h,spreads,totals
#   - Player Props: /sports/{sport_key}/events then /events/{event_id}/odds (ONE market per request)
# ✅ Best price + EV + Auto-ranked Top 2–5
# ✅ Dynamic prop market discovery (cached + snapshot)
# ✅ No uploads, no Kelly, deploy-ready

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
.block-container {padding-top: 1rem;}
section[data-testid="stSidebar"] {background: #0b1220;}
section[data-testid="stSidebar"] * {color: #e8eefc;}
h1,h2,h3 {letter-spacing:.2px;}
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px; border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎯 EdgeLedger Dashboard")

# -----------------------------
# API CONFIG (YOUR KEY)
# -----------------------------
API_KEY = "6a5d08e7c2407da6fb95b86ad9619bf0"  # <-- Provided key

BASE_URL = "https://api.the-odds-api.com/v4"
REGION = "us"
ODDS_FORMAT = "american"
HEADERS = {"User-Agent": "EdgeLedgerDashboard/3.0"}

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
GAME_LINES_TTL = 15 * 60    # 15 minutes
EVENTS_TTL = 30 * 60        # 30 minutes
PROPS_TTL = 15 * 60         # 15 minutes
MARKETS_TTL = 24 * 60 * 60  # 24 hours

# Require click to fetch from API (prevents rerun credit burn)
REQUIRE_CLICK_TO_LOAD = True

# Snapshot folder (works on Streamlit Cloud; may reset on redeploy, but keeps app operational)
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
    """
    Best price across books for American odds = numeric max
    (+200 > +180, -105 > -115)
    """
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
    """HTTP GET with robust error handling; returns (ok, payload, status_code)."""
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        status = r.status_code
        try:
            payload = r.json()
        except Exception:
            payload = {"message": r.text}

        if status == 200:
            return True, payload, status

        # Explicitly preserve quota payload for UI handling
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
    """Normalize /events/{event_id}/odds response for ONE prop market into player-based rows."""
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
def api_fetch_events(sport_key: str):
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
        "markets": market_key,  # ONE market per call (prevents 422)
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
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Dashboard")
sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_name]

view = st.sidebar.radio("View", ["Game Lines", "Player Props"], index=0)
debug_mode = st.sidebar.toggle("Show Debug", value=False)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Auto-rank Top Bets (by EV)", 2, 5, 3, 1)

# Click-to-load prevents rerun credit burn
load_clicked = True
if REQUIRE_CLICK_TO_LOAD:
    load_clicked = st.sidebar.button("📥 Load / Refresh Data")

if st.sidebar.button("🔄 Force refresh (clear cache)"):
    st.cache_data.clear()
    st.rerun()

# -----------------------------
# HEADER METRICS
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Sport", sport_name)
c2.metric("Last Refresh", now_utc())
c3.metric("Mode", view)

tabs = st.tabs(["Dashboard", "All Books (Raw)", "Diagnostics / Help"])

# If not clicked, show snapshots if available (operational site even without credits)
if REQUIRE_CLICK_TO_LOAD and not load_clicked:
    st.info("Click **📥 Load / Refresh Data** in the sidebar to fetch new odds (prevents burning credits).")
    st.caption("Showing last saved snapshot if available.")

# -----------------------------
# GAME LINES (separate API call + snapshot fallback)
# -----------------------------
if view == "Game Lines":
    chosen_human = st.sidebar.multiselect(
        "Game Line Markets",
        list(GAME_MARKETS.keys()),
        default=["Spreads", "Totals", "Moneyline (H2H)"],
    )
    chosen = [GAME_MARKETS[x] for x in chosen_human] if chosen_human else ["h2h", "spreads", "totals"]

    snap_df_path = snap_path(f"lines_{sport_key}.parquet")
    snap_dbg_path = snap_path(f"lines_dbg_{sport_key}.json")

    df_lines = pd.DataFrame()
    dbg_lines: Dict[str, Any] = {}

    # If user clicked (or click not required), attempt API fetch
    if (not REQUIRE_CLICK_TO_LOAD) or load_clicked:
        df_lines, dbg_lines = api_fetch_game_lines_with_fallback(sport_key, chosen)

        # If out of credits, fallback to snapshot immediately
        if (not dbg_lines.get("ok", True)) and is_out_of_credits(dbg_lines.get("status", 0), dbg_lines.get("error")):
            st.warning("🚫 Odds API credits exhausted (OUT_OF_USAGE_CREDITS). Showing last snapshot if available.")
            df_lines = load_snapshot_df(snap_df_path)
            old_dbg = load_snapshot_json(snap_dbg_path) or {}
            dbg_lines["snapshot_used"] = True
            dbg_lines["snapshot_dbg"] = old_dbg

        # If we got fresh rows, snapshot them
        if not df_lines.empty:
            save_snapshot_df(snap_df_path, df_lines)
            save_snapshot_json(snap_dbg_path, dbg_lines)

    # If no API fetch happened (no click) OR fetch yielded nothing, show snapshot
    if df_lines.empty:
        df_lines = load_snapshot_df(snap_df_path)
        if not dbg_lines:
            dbg_lines = load_snapshot_json(snap_dbg_path) or {}
            dbg_lines["snapshot_used"] = True

    # Final empty handling
    if df_lines.empty:
        st.warning("No game line data available right now (and no saved snapshot yet).")
        with st.expander("Diagnostics (game lines)"):
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
    top.insert(0, "⭐", "⭐")  # highlight that these are top-ranked

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
        st.write(dbg_lines)

# -----------------------------
# PLAYER PROPS (separate API flow + snapshot fallback)
# -----------------------------
else:
    snap_best_path = snap_path(f"props_best_{sport_key}.parquet")
    snap_raw_path = snap_path(f"props_raw_{sport_key}.parquet")
    snap_dbg_path = snap_path(f"props_dbg_{sport_key}.json")
    snap_markets_path = snap_path(f"markets_{sport_key}.json")

    markets_list: List[dict] = []
    dbg_mk: Dict[str, Any] = {}

    # Discover markets (cached); if out of credits, use saved snapshot
    if (not REQUIRE_CLICK_TO_LOAD) or load_clicked:
        markets_list, dbg_mk = api_discover_markets(sport_key)

        if (not dbg_mk.get("ok", True)) and is_out_of_credits(dbg_mk.get("status", 0), dbg_mk.get("error")):
            st.warning("🚫 Credits exhausted during market discovery. Using last saved markets list if available.")
            markets_list = load_snapshot_json(snap_markets_path) or []
            dbg_mk["snapshot_used"] = True

        if markets_list:
            save_snapshot_json(snap_markets_path, markets_list)

    # If no click / empty, use saved markets
    if not markets_list:
        markets_list = load_snapshot_json(snap_markets_path) or []

    discovered = {}
    for m in markets_list:
        if isinstance(m, dict):
            k = m.get("key") or m.get("market_key")
            n = m.get("name") or m.get("title") or pretty_market_name(k)
            if k:
                discovered[k] = n

    player_markets = sorted([k for k in discovered.keys() if k.startswith("player_") and k != "player_props"])
    if not player_markets:
        st.warning("No player prop markets are available (and no saved snapshot yet).")
        with tabs[2]:
            st.subheader("Diagnostics (Market Discovery)")
            st.write(dbg_mk)
        st.stop()

    # Dropdown for prop market
    prop_market_key = st.sidebar.selectbox(
        "Player Prop Market (auto-discovered)",
        options=player_markets,
        format_func=lambda k: discovered.get(k, pretty_market_name(k)),
        index=0,
    )

    # If no click, show snapshot tables if available
    if REQUIRE_CLICK_TO_LOAD and not load_clicked:
        best_snap = load_snapshot_df(snap_best_path)
        raw_snap = load_snapshot_df(snap_raw_path)
        dbg_snap = load_snapshot_json(snap_dbg_path) or {"snapshot_used": True}

        if best_snap.empty:
            st.info("Click **📥 Load / Refresh Data** to fetch props. No saved snapshot yet for props.")
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
            st.write(dbg_snap)

        st.stop()

    # Otherwise, fetch props using separate calls: events -> event odds
    events, dbg_ev = api_fetch_events(sport_key)

    # Handle out-of-credits
    if (not dbg_ev.get("ok", True)) and is_out_of_credits(dbg_ev.get("status", 0), dbg_ev.get("error")):
        st.warning("🚫 Credits exhausted while loading events. Showing last props snapshot if available.")
        best_snap = load_snapshot_df(snap_best_path)
        raw_snap = load_snapshot_df(snap_raw_path)
        dbg_snap = load_snapshot_json(snap_dbg_path) or {}
        dbg_snap["snapshot_used"] = True

        if best_snap.empty:
            st.info("No saved props snapshot exists yet. Upgrade/wait for quota reset then click Load.")
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
            st.write({"events_dbg": dbg_ev, "snapshot_dbg": dbg_snap})
        st.stop()

    if not events:
        st.warning("No events returned for this sport right now.")
        with tabs[2]:
            st.subheader("Diagnostics (Events)")
            st.write(dbg_ev)
        st.stop()

    all_props = []
    prop_diag = {"attempted_events": 0, "events_with_rows": 0, "skipped_empty": 0, "errors": 0}

    with st.spinner("Loading player props (event-by-event)…"):
        max_events_to_try = min(20, len(events))  # reduce calls to protect credits
        for e in events[:max_events_to_try]:
            event_id = e.get("id")
            if not event_id:
                prop_diag["skipped_empty"] += 1
                continue

            prop_diag["attempted_events"] += 1
            df_evprops, dbg_p = api_fetch_event_props(sport_key, event_id, prop_market_key)

            if not dbg_p.get("ok", True):
                prop_diag["errors"] += 1
                # if out of credits mid-loop, stop early and use what we have
                if is_out_of_credits(dbg_p.get("status", 0), dbg_p.get("error")):
                    break

            if df_evprops.empty:
                prop_diag["skipped_empty"] += 1
                continue

            prop_diag["events_with_rows"] += 1
            all_props.append(df_evprops)

    if not all_props:
        st.warning("No props rows were normalized for that market right now.")
        st.info("Try another market from the dropdown (availability varies by sport/date/book).")
        with tabs[2]:
            st.subheader("Diagnostics (Props)")
            st.write({"events_dbg": dbg_ev, "prop_diag": prop_diag})
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

    # Save snapshots (operational fallback)
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
        st.write({"market_discovery_dbg": dbg_mk, "events_dbg": dbg_ev, "prop_diag": prop_diag})

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

### Why this stays operational
- The app only fetches data when you click **📥 Load / Refresh Data**
- It caches results aggressively
- It saves snapshots and shows the last snapshot when credits are exhausted

### If you see OUT_OF_USAGE_CREDITS
Upgrade your Odds API plan or wait for quota reset, then click **Load / Refresh Data**.
"""
    )
