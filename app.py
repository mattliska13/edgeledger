# app.py — EdgeLedger Dashboard (NFL / CFB)
# ✅ Game Lines + Player Props with completely separate API paths (prevents 422)
# ✅ Auto-discover prop markets per sport (dynamic dropdown)
# ✅ NO Kelly / bankroll sizing anywhere
# ✅ Best price across books + EV (using consensus implied baseline)
# ✅ Auto-ranked Top 2–5 bets
# ✅ Robust empty checks + ALWAYS-visible API diagnostics (fixes “No data” mystery)
# ✅ Force Refresh button (clears cache + reruns)
# ✅ No upload dependencies
# ✅ Fixed triple-quoted markdown (no SyntaxError)

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

        # If bookmakers empty, skip gracefully
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
# API CALLS (SEPARATED) + DIAGNOSTICS
# -----------------------------
@st.cache_data(ttl=120, show_spinner=False)
def api_discover_markets(sport_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/markets"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "payload_type": type(payload).__name__}
    if ok and isinstance(payload, list):
        debug["payload_len"] = len(payload)
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

    debug = {
        "url": url,
        "status": status,
        "ok": ok,
        "markets": markets,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
        "error": None if ok else payload,
        "sample": payload[0] if ok and isinstance(payload, list) and payload else None,
    }

    if ok and isinstance(payload, list):
        df = normalize_game_lines(payload)
        debug["rows_normalized"] = int(df.shape[0])
        return df, debug

    debug["rows_normalized"] = 0
    return pd.DataFrame(), debug


@st.cache_data(ttl=90, show_spinner=False)
def api_fetch_events(sport_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {"apiKey": API_KEY}
    ok, payload, status = safe_get(url, params=params)
    debug = {
        "url": url,
        "status": status,
        "ok": ok,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
        "error": None if ok else payload,
        "sample": payload[0] if ok and isinstance(payload, list) and payload else None,
    }
    if ok and isinstance(payload, list):
        return payload, debug
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
    debug = {"url": url, "status": status, "ok": ok, "market": market_key, "error": None if ok else payload}
    if ok and isinstance(payload, dict):
        df = normalize_event_props(payload, market_key)
        debug["rows_normalized"] = int(df.shape[0])
        return df, debug
    debug["rows_normalized"] = 0
    return pd.DataFrame(), debug


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Dashboard")
sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_name]

view = st.sidebar.radio("View", ["Game Lines", "Player Props"], index=0)
debug_mode = st.sidebar.toggle("Show Debug", value=False)

# 🔄 Force refresh button
if st.sidebar.button("🔄 Force refresh (clear cache)"):
    st.cache_data.clear()
    st.rerun()

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

    # ✅ Bulletproof default if user unselects everything
    if not chosen_human:
        chosen = ["h2h", "spreads", "totals"]
    else:
        chosen = [GAME_MARKETS[x] for x in chosen_human]

    df_lines, dbg_lines = api_fetch_game_lines(sport_key, chosen)

    if debug_mode:
        st.sidebar.markdown("**DEBUG (Game Lines)**")
        st.sidebar.write(dbg_lines)

    if df_lines.empty:
        st.warning("No game line rows were normalized from the API response.")
        with st.expander("Show API diagnostics (game lines)"):
            st.write(dbg_lines)

        st.info(
            "Most common causes:\n"
            "- No events with odds available for this sport right now\n"
            "- Events returned but bookmakers list is empty\n"
            "- API returned an error (rate limit / bad params / plan restriction)"
        )
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
            st.subheader("🔥 Auto-Ranked Top Bets (Game Lines)")
            st.caption("Best price highlights the single best bookmaker per line. EV uses consensus implied baseline (no external model).")
            st.dataframe(
                top[["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
                use_container_width=True,
            )

            st.subheader("Best Price Board (Game Lines)")
            st.dataframe(
                best[["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
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

    discovered = {}
    for m in markets_list:
        if isinstance(m, dict):
            k = m.get("key") or m.get("market_key")
            n = m.get("name") or m.get("title") or pretty_market_name(k)
            if k:
                discovered[k] = n

    player_markets = sorted([k for k in discovered.keys() if k.startswith("player_") and k != "player_props"])
    if not player_markets:
        st.warning("No player prop markets discovered for this sport on your plan/books right now.")
        with st.expander("Show API diagnostics (market discovery)"):
            st.write(dbg_mk)
        st.stop()

    prop_market_key = st.sidebar.selectbox(
        "Player Prop Market (auto-discovered)",
        options=player_markets,
        format_func=lambda k: discovered.get(k, pretty_market_name(k)),
        index=0,
    )

    events, dbg_ev = api_fetch_events(sport_key)

    if debug_mode:
        st.sidebar.markdown("**DEBUG (Events)**")
        st.sidebar.write(dbg_ev)

    if not events:
        st.warning("No events returned for this sport right now.")
        with st.expander("Show API diagnostics (events)"):
            st.write(dbg_ev)
        st.stop()

    all_props = []
    prop_diag = {"attempted_events": 0, "events_with_rows": 0, "skipped_empty": 0, "errors": 0}

    with st.spinner("Loading player props (event-by-event)…"):
        max_events_to_try = min(25, len(events))
        for e in events[:max_events_to_try]:
            event_id = e.get("id")
            if not event_id:
                prop_diag["skipped_empty"] += 1
                continue
            prop_diag["attempted_events"] += 1

            df_evprops, dbg_p = api_fetch_event_props(sport_key, event_id, prop_market_key)

            if not dbg_p.get("ok", True):
                prop_diag["errors"] += 1

            if df_evprops.empty:
                prop_diag["skipped_empty"] += 1
                continue

            prop_diag["events_with_rows"] += 1
            all_props.append(df_evprops)

    if not all_props:
        st.warning("No props rows were normalized for that market right now.")
        with st.expander("Show Prop Diagnostics"):
            st.write(prop_diag)
        st.info("Try another prop market from the dropdown — availability changes by week/book.")
        st.stop()

    raw_props = pd.concat(all_props, ignore_index=True)
    raw_props["Player"] = raw_props["Player"].fillna("").astype(str).str.strip()
    raw_props = raw_props[raw_props["Player"].str.len() > 0].copy()

    if raw_props.empty:
        st.warning("Props returned, but player fields were empty/unusable.")
        with st.expander("Show Prop Diagnostics"):
            st.write(prop_diag)
        st.stop()

    raw_props["PosBucket"] = raw_props["Market"].apply(infer_position_bucket)

    group_cols = ["Event", "Market", "Player", "Side", "Line"]
    cons = raw_props.groupby(group_cols).apply(consensus_true_prob).rename("P_base").reset_index()

    best = best_price_table(raw_props, group_cols, "Price")
    best["Best Implied"] = best["Best Price"].apply(american_to_implied)

    best = best.merge(cons, on=group_cols, how="left")
    best["PosBucket"] = best["Market"].apply(infer_position_bucket)
    best["P_used"] = best.apply(lambda r: apply_bucket_adjustment(r["P_base"], r["PosBucket"], r["Market"]), axis=1)

    best["EV"] = (best["P_used"] - best["Best Implied"]) * 100.0

    best = best.sort_values("EV", ascending=False)
    top = best.head(top_n).copy()

    with tabs[0]:
        st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {discovered.get(prop_market_key, pretty_market_name(prop_market_key))}")
        st.caption("Best price highlights the single best bookmaker per prop. EV uses consensus implied baseline + small position-bucket adjustment.")
        st.dataframe(
            top[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV"]],
            use_container_width=True,
        )

        st.subheader("Best Price Board (Player Props)")
        st.dataframe(
            best[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV"]],
            use_container_width=True,
        )

        with st.expander("Show Prop Diagnostics"):
            st.write(prop_diag)

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
