# app.py — EdgeLedger Dashboard (NFL / CFB)
# ✅ Uses your provided API key (hard-coded) for now
# ✅ Game Lines and Player Props use completely separate API calls (prevents cross-talk + 422)
# ✅ Game lines: /sports/{sport_key}/odds (markets=h2h,spreads,totals)
# ✅ Player props: /sports/{sport_key}/events -> /events/{event_id}/odds (ONE market per request)
# ✅ Dynamic prop market discovery (/sports/{sport_key}/markets) to populate dropdown
# ✅ Best price across books + EV (consensus implied baseline)
# ✅ Auto-ranked Top 2–5 bets
# ✅ Robust empty checks + ALWAYS shows diagnostics when empty
# ✅ Force refresh (clears cache)
# ✅ No uploads, no Kelly, deploy-ready

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone

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
HEADERS = {"User-Agent": "EdgeLedgerDashboard/2.0"}

SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

GAME_MARKETS = {
    "Moneyline (H2H)": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

# Some accounts return better bookmaker coverage when filtered.
FALLBACK_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "pinnacle", "circa"]

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
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def consensus_true_prob(df_group: pd.DataFrame) -> float:
    """Baseline probability from market (average implied across books)."""
    vals = df_group["Implied"].dropna().values
    if len(vals) == 0:
        return np.nan
    return float(np.clip(vals.mean(), 0.02, 0.98))


def pretty_market_name(k: str) -> str:
    return (k or "").replace("_", " ").title()


def best_price_table(df: pd.DataFrame, group_cols: list[str], price_col: str = "Price") -> pd.DataFrame:
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
# API CALLS (SEPARATED)
# -----------------------------
@st.cache_data(ttl=120, show_spinner=False)
def api_discover_markets(sport_key: str):
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
        "sample": payload[0] if ok and isinstance(payload, list) and payload else None,
    }
    return payload if ok and isinstance(payload, list) else [], dbg


@st.cache_data(ttl=60, show_spinner=False)
def api_fetch_game_lines_once(sport_key: str, markets: list[str], bookmakers: list[str] | None = None):
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
        "sample": payload[0] if ok and isinstance(payload, list) and payload else None,
    }

    if ok and isinstance(payload, list):
        df = normalize_game_lines(payload)
        dbg["rows_normalized"] = int(df.shape[0])
        return df, dbg

    dbg["rows_normalized"] = 0
    return pd.DataFrame(), dbg


def api_fetch_game_lines_with_fallback(sport_key: str, markets: list[str]):
    df1, dbg1 = api_fetch_game_lines_once(sport_key, markets, bookmakers=None)
    if not df1.empty:
        dbg1["fallback_used"] = False
        return df1, dbg1

    df2, dbg2 = api_fetch_game_lines_once(sport_key, markets, bookmakers=FALLBACK_BOOKMAKERS)
    dbg2["fallback_used"] = True
    dbg2["first_attempt"] = dbg1
    return df2, dbg2


@st.cache_data(ttl=90, show_spinner=False)
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
        "sample": payload[0] if ok and isinstance(payload, list) and payload else None,
    }
    return payload if ok and isinstance(payload, list) else [], dbg


@st.cache_data(ttl=60, show_spinner=False)
def api_fetch_event_props(sport_key: str, event_id: str, market_key: str):
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": market_key,  # ONE market per call
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
    }
    if ok and isinstance(payload, dict):
        df = normalize_event_props(payload, market_key)
        dbg["rows_normalized"] = int(df.shape[0])
        return df, dbg
    dbg["rows_normalized"] = 0
    return pd.DataFrame(), dbg


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Dashboard")
sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_name]

view = st.sidebar.radio("View", ["Game Lines", "Player Props"], index=0)
debug_mode = st.sidebar.toggle("Show Debug", value=False)

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

tabs = st.tabs(["Dashboard", "All Books (Raw)", "Diagnostics / Help"])

# -----------------------------
# GAME LINES (separate API call)
# -----------------------------
if view == "Game Lines":
    chosen_human = st.sidebar.multiselect(
        "Game Line Markets",
        list(GAME_MARKETS.keys()),
        default=["Spreads", "Totals", "Moneyline (H2H)"],
    )
    chosen = [GAME_MARKETS[x] for x in chosen_human] if chosen_human else ["h2h", "spreads", "totals"]

    df_lines, dbg_lines = api_fetch_game_lines_with_fallback(sport_key, chosen)

    if df_lines.empty:
        st.warning("No game line rows were normalized from the API response.")
        with st.expander("Show API diagnostics (game lines)"):
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
    top = best.head(top_n)

    with tabs[0]:
        st.subheader("🔥 Auto-Ranked Top Bets (Game Lines)")
        st.caption("Best price = best bookmaker for that exact line/outcome. EV uses consensus implied baseline.")
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
        st.dataframe(raw[["Event", "Market", "Outcome", "Line", "Price", "Book", "Implied"]], use_container_width=True)

    if debug_mode:
        with tabs[2]:
            st.subheader("Diagnostics (Game Lines)")
            st.write(dbg_lines)

# -----------------------------
# PLAYER PROPS (separate API calls: markets -> events -> event odds)
# -----------------------------
else:
    markets_list, dbg_mk = api_discover_markets(sport_key)

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
        st.info("Try another market from the dropdown (availability varies by week/book).")
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
    top = best.head(top_n)

    with tabs[0]:
        st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {discovered.get(prop_market_key, pretty_market_name(prop_market_key))}")
        st.caption("Best price = best bookmaker per player/side/line. EV uses consensus implied baseline.")
        st.dataframe(
            top[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )
        st.subheader("Best Price Board (Player Props)")
        st.dataframe(
            best[["Event", "Player", "Side", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV"]],
            use_container_width=True,
        )
        with st.expander("Show Prop Diagnostics"):
            st.write(prop_diag)

    with tabs[1]:
        st.subheader("All Books (Raw Player Props)")
        st.dataframe(
            raw_props[["Event", "Market", "Player", "Side", "Line", "Price", "Book", "Implied"]],
            use_container_width=True,
        )

    if debug_mode:
        with tabs[2]:
            st.subheader("Diagnostics (Player Props)")
            st.write({"market_discovery": dbg_mk, "events": dbg_ev, "prop_diag": prop_diag})

# -----------------------------
# DEPLOY HELP
# -----------------------------
with tabs[2]:
    st.subheader("Deploy Notes (Operational Site)")
    st.markdown(
        """
**Streamlit Cloud:**  
1) Push this `app.py` to GitHub  
2) Streamlit Cloud → New app → Select repo → Main file: `app.py`  
3) Deploy

**If Game Lines show empty:** Open “Show API diagnostics (game lines)” and look at:
- `status` (200 vs 401/429)
- `payload_len`
- `rows_normalized`
- `fallback_used`

**If Props show empty:** Try a different prop market — availability varies by books + sport + date.
"""
    )
