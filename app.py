# app.py — EdgeLedger Dashboard (NFL / CFB)
# ✅ Game Lines + Player Props with completely separate API paths (prevents 422)
# ✅ Auto-discover prop markets per sport (dynamic dropdown)
# ✅ NO Kelly / bankroll sizing anywhere
# ✅ Best price across books + EV (using consensus implied baseline)
# ✅ Auto-ranked Top 2–5 bets
# ✅ Robust empty checks + debug logging
# ✅ No upload dependencies

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# UI / THEME
# -----------------------------
st.set_page_config(page_title="EdgeLedger Dashboard", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {padding-top: 1.0rem;}
section[data-testid="stSidebar"] {background: #0b1220;}
section[data-testid="stSidebar"] * {color: #e8eefc;}
h1,h2,h3 {letter-spacing:.2px;}
div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  padding: 10px 12px; border-radius: 14px;
}
.small-note {opacity:.85; font-size: 0.92rem;}
.badge {
  display:inline-block; padding: 2px 8px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.05);
  font-size: 0.85rem; margin-left: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎯 EdgeLedger Dashboard")

# -----------------------------
# SECRETS / CONFIG
# -----------------------------
API_KEY = st.secrets.get("ODDS_API_KEY", "") or os.getenv("ODDS_API_KEY", "")
if not API_KEY:
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Cloud → Settings → Secrets:\n\nODDS_API_KEY="xxxx"')
    st.stop()

BASE_URL = "https://api.the-odds-api.com/v4"
REGION = "us"
ODDS_FORMAT = "american"
HEADERS = {"User-Agent": "EdgeLedgerDashboard/1.0"}

SPORTS = {"NFL": "americanfootball_nfl", "CFB": "americanfootball_ncaaf"}
GAME_MARKETS = {"Moneyline (H2H)": "h2h", "Spreads": "spreads", "Totals": "totals"}

# -----------------------------
# HELPERS
# -----------------------------
def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def safe_get(url: str, params: dict, timeout: int = 20):
    """HTTP GET with robust error handling; returns (ok, payload, status_code)."""
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
        if r.status_code == 200:
            return True, r.json(), 200
        try:
            payload = r.json()
        except Exception:
            payload = {"message": r.text}
        return False, payload, r.status_code
    except Exception as e:
        return False, {"message": str(e)}, -1


def american_to_implied(odds):
    """American odds -> implied prob."""
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def pretty_market_name(mkey: str) -> str:
    return (mkey or "").replace("_", " ").title()


def infer_position_bucket(market_key: str) -> str:
    mk = (market_key or "").lower()
    if "pass" in mk:
        return "QB"
    if "rush" in mk:
        return "RB"
    if "rec" in mk or "recept" in mk:
        return "WR/TE"
    if "td" in mk:
        return "SKILL"
    return "UNKNOWN"


def consensus_true_prob(df_group: pd.DataFrame) -> float:
    """
    Baseline “real” probability without projections:
    average implied probability across books for the exact selection.
    """
    vals = df_group["Implied"].dropna().values
    if len(vals) == 0:
        return np.nan
    return float(np.clip(vals.mean(), 0.02, 0.98))


def apply_bucket_adjustment(p_base: float, bucket: str, market_key: str) -> float:
    """
    Small conservative adjustment to break ties (QB vs RB vs WR/TE) without external data.
    """
    if np.isnan(p_base):
        return np.nan
    mk = (market_key or "").lower()
    b = (bucket or "").upper()
    adj = 0.0

    # TD props: slightly favor RB / WR/TE, slightly fade QB (conservative)
    if "anytime" in mk or "td" in mk:
        if b == "RB":
            adj += 0.010
        elif b in ("WR/TE", "SKILL"):
            adj += 0.006
        elif b == "QB":
            adj -= 0.005

    # Passing yard overs tend to be sharper; tiny regression
    if ("pass" in mk) and ("yd" in mk):
        adj -= 0.004

    return float(np.clip(p_base + adj, 0.02, 0.98))


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
                        dict(
                            Event=event_name,
                            Commence=commence,
                            Market=mkey,
                            Outcome=out.get("name"),
                            Line=out.get("point"),
                            Price=out.get("price"),
                            Book=book,
                        )
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)
    return df


def normalize_event_props(event_odds_json: dict, market_key: str) -> pd.DataFrame:
    """
    Normalizes /events/{event_id}/odds response for ONE prop market into player-based rows.
    """
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
                side = (out.get("name") or "").strip()
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
                    dict(
                        Event=event_name,
                        Commence=commence,
                        Market=market_key,
                        Player=player,
                        Side=side,
                        Line=out.get("point"),
                        Price=out.get("price"),
                        Book=book,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)
    df["Side"] = df["Side"].fillna("").astype(str)
    df["Player"] = df["Player"].fillna("").astype(str).str.strip()
    return df


def best_price_table(df: pd.DataFrame, group_cols: list[str], price_col: str = "Price") -> pd.DataFrame:
    """
    Best price across books for American odds = numeric max
    (+200 > +180, -105 > -115).
    """
    if df.empty:
        return df
    idx = df.groupby(group_cols)[price_col].idxmax()
    best = df.loc[idx].copy()
    best = best.rename(columns={price_col: "Best Price", "Book": "Best Book"})
    return best.reset_index(drop=True)


# -----------------------------
# API CALLS (SEPARATED)
# -----------------------------
@st.cache_data(ttl=120, show_spinner=False)
def api_discover_markets(sport_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/markets"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "error": None}
    if ok and isinstance(payload, list):
        return payload, debug
    debug["error"] = payload
    return [], debug


@st.cache_data(ttl=60, show_spinner=False)
def api_fetch_game_lines(sport_key: str, markets: list[str]):
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
    }
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "error": None, "markets": markets, "sample": None}
    if ok and isinstance(payload, list):
        debug["sample"] = payload[0] if payload else None
        return normalize_game_lines(payload), debug
    debug["error"] = payload
    return pd.DataFrame(), debug


@st.cache_data(ttl=90, show_spinner=False)
def api_fetch_events(sport_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "error": None, "sample": None}
    if ok and isinstance(payload, list):
        debug["sample"] = payload[0] if payload else None
        return payload, debug
    debug["error"] = payload
    return [], debug


@st.cache_data(ttl=60, show_spinner=False)
def api_fetch_event_props(sport_key: str, event_id: str, market_key: str):
    # ONE market per request => avoids 422
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": ODDS_FORMAT,
    }
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "error": None, "market": market_key}
    if ok and isinstance(payload, dict):
        return normalize_event_props(payload, market_key), debug
    debug["error"] = payload
    return pd.DataFrame(), debug


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

# -----------------------------
# HEADER METRICS
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Sport", sport_name)
c2.metric("Last Refresh", now_utc())
c3.metric("Mode", view)

tabs = st.tabs(["Dashboard", "All Books (Raw)", "Setup Help"])

# -----------------------------
# GAME LINES
# -----------------------------
if view == "Game Lines":
    chosen_human = st.sidebar.multiselect(
        "Game Line Markets",
        list(GAME_MARKETS.keys()),
        default=["Spreads", "Totals", "Moneyline (H2H)"],
    )

    # ✅ Bulletproof default if user unselects everything (fixes "No game line data..." due to empty markets)
    if not chosen_human:
        chosen = ["h2h", "spreads", "totals"]
    else:
        chosen = [GAME_MARKETS[x] for x in chosen_human]

    df_lines, dbg_lines = api_fetch_game_lines(sport_key, chosen)

    if debug_mode:
        st.sidebar.markdown("**DEBUG (Game Lines)**")
        st.sidebar.write(dbg_lines)

    if df_lines.empty:
        st.warning("No game line data available right now for these settings (or no upcoming games returned).")
        if debug_mode:
            st.info("Turn on Debug and check status/error payload in the sidebar.")
    else:
        raw = df_lines.copy()

        # Best price per event/market/outcome/line
        best = best_price_table(raw, ["Event", "Market", "Outcome", "Line"], "Price")
        best["Best Implied"] = best["Best Price"].apply(american_to_implied)

        # Consensus baseline p_true per selection across books
        cons = (
            raw.groupby(["Event", "Market", "Outcome", "Line"])
            .apply(consensus_true_prob)
            .rename("P_true")
            .reset_index()
        )
        best = best.merge(cons, on=["Event", "Market", "Outcome", "Line"], how="left")

        # EV
        best["EV"] = (best["P_true"] - best["Best Implied"]) * 100.0

        best = best.sort_values("EV", ascending=False)
        top = best.head(top_n).copy()

        with tabs[0]:
            st.subheader("🔥 Auto-Ranked Top Bets (Game Lines)  ")
            st.caption("Best price highlights the single best bookmaker per line. EV uses consensus implied baseline (no external model).")
            st.dataframe(
                top[
                    ["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]
                ],
                use_container_width=True,
            )

            st.subheader("Best Price Board (Game Lines)")
            st.dataframe(
                best[
                    ["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]
                ],
                use_container_width=True,
            )

        with tabs[1]:
            st.subheader("All Books (Raw Game Lines)")
            st.dataframe(
                raw[["Event", "Market", "Outcome", "Line", "Price", "Book", "Implied"]],
                use_container_width=True,
            )

# -----------------------------
# PLAYER PROPS
# -----------------------------
else:
    markets_list, dbg_mk = api_discover_markets(sport_key)

    if debug_mode:
        st.sidebar.markdown("**DEBUG (Market Discovery)**")
        st.sidebar.write(dbg_mk)

    # Build key -> name mapping
    discovered = {}
    for m in markets_list:
        if isinstance(m, dict):
            k = m.get("key") or m.get("market_key")
            n = m.get("name") or m.get("title") or pretty_market_name(k)
            if k:
                discovered[k] = n

    # Filter likely player props
    player_markets = sorted([k for k in discovered.keys() if k.startswith("player_") and k != "player_props"])
    if not player_markets:
        st.warning("No player prop markets discovered for this sport on your plan/books right now.")
        st.stop()

    prop_market_key = st.sidebar.selectbox(
        "Player Prop Market (auto-discovered)",
        options=player_markets,
        format_func=lambda k: discovered.get(k, pretty_market_name(k)),
        index=0,
    )

    # Fetch events (separate endpoint)
    events, dbg_ev = api_fetch_events(sport_key)

    if debug_mode:
        st.sidebar.markdown("**DEBUG (Events)**")
        st.sidebar.write(dbg_ev)

    if not events:
        st.warning("No events returned for this sport right now.")
        st.stop()

    all_props = []
    failures = []

    with st.spinner("Loading player props (event-by-event)…"):
        max_events_to_try = min(25, len(events))
        tried = 0

        for e in events:
            if tried >= max_events_to_try:
                break
            event_id = e.get("id")
            if not event_id:
                continue
            tried += 1

            df_evprops, dbg_p = api_fetch_event_props(sport_key, event_id, prop_market_key)
            if df_evprops.empty:
                if debug_mode and not dbg_p.get("ok", True):
                    failures.append(dbg_p)
                continue
            all_props.append(df_evprops)

    if debug_mode and failures:
        st.sidebar.markdown("**DEBUG (Prop Failures / Skipped)**")
        st.sidebar.write(failures[:5])

    if not all_props:
        st.warning("No props returned for that market right now. Try a different market in the dropdown.")
        st.stop()

    raw_props = pd.concat(all_props, ignore_index=True)
    raw_props["Player"] = raw_props["Player"].fillna("").astype(str).str.strip()
    raw_props = raw_props[raw_props["Player"].str.len() > 0].copy()

    if raw_props.empty:
        st.warning("Props returned, but player fields were empty/unusable.")
        st.stop()

    raw_props["PosBucket"] = raw_props["Market"].apply(infer_position_bucket)

    # Baseline p_base per player/side/line selection (across books)
    group_cols = ["Event", "Market", "Player", "Side", "Line"]
    cons = raw_props.groupby(group_cols).apply(consensus_true_prob).rename("P_base").reset_index()

    # Best price per selection across books
    best = best_price_table(raw_props, group_cols, "Price")
    best["Best Implied"] = best["Best Price"].apply(american_to_implied)

    # Merge base prob, apply bucket adjustment
    best = best.merge(cons, on=group_cols, how="left")
    best["PosBucket"] = best["Market"].apply(infer_position_bucket)
    best["P_used"] = best.apply(lambda r: apply_bucket_adjustment(r["P_base"], r["PosBucket"], r["Market"]), axis=1)

    # EV
    best["EV"] = (best["P_used"] - best["Best Implied"]) * 100.0

    best = best.sort_values("EV", ascending=False)
    top = best.head(top_n).copy()

    with tabs[0]:
        st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {discovered.get(prop_market_key, pretty_market_name(prop_market_key))}")
        st.caption("Best price highlights the single best bookmaker per prop. EV uses consensus implied baseline + small position-bucket adjustment.")
        st.dataframe(
            top[
                ["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV"]
            ],
            use_container_width=True,
        )

        st.subheader("Best Price Board (Player Props)")
        st.dataframe(
            best[
                ["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV"]
            ],
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("All Books (Raw Player Props)")
        st.dataframe(
            raw_props[["Event", "Market", "Player", "Side", "Line", "Price", "Book", "Implied", "PosBucket"]],
            use_container_width=True,
        )

# -----------------------------
# SETUP HELP (FIXED triple-quote)
# -----------------------------
with tabs[2]:
    st.subheader("Setup Help")
    st.markdown(
        """
### 1) Auto-discover markets (dynamic dropdown)
This app pulls available markets for your sport/plan from:

`GET /v4/sports/{sport_key}/markets`

So you always see **only what your account supports**.

---

### 2) Separate API call paths (prevents 422 errors)
**Game Lines**
- `GET /v4/sports/{sport_key}/odds?markets=h2h,spreads,totals`

**Player Props**
1. `GET /v4/sports/{sport_key}/events`
2. `GET /v4/sports/{sport_key}/events/{event_id}/odds?markets=<ONE_MARKET_KEY>`

Props are fetched **event-by-event** and **one market at a time**, which avoids invalid market combinations.

---

### 3) EV and ranking without uploads
Because you requested **no dependency on uploads**, the app uses:
- Baseline true probability = **average implied probability across books** for the same selection
- Small position-bucket adjustment (QB vs RB vs WR/TE) to add conservative signal

This keeps EV ranking stable without external datasets.
"""
    )
