# app.py — EdgeLedger Dashboard (NFL/CFB)
# ✅ Separate API calls: Game Lines vs Player Props (event-by-event, 1 market per request)
# ✅ Auto-discover available markets per sport and populate dropdown dynamically
# ✅ Prop-specific modeling (QB vs RB vs WR/TE) with real projections (CSV-supported)
# ✅ Kelly bankroll sizing
# ✅ Best price + EV + Top 2–5 auto-ranked bets
#
# NOTES:
# - Your plan/books determine which prop markets appear. We discover them live.
# - “Real projections” means: you can upload your own projections (recommended).
#   If you don't upload projections, we fall back to a conservative market-implied baseline
#   (consensus implied prob across books after normalization), so EV still works.

import os
import math
import json
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


def american_to_decimal(odds):
    o = float(odds)
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def norm_cdf(x):
    # Standard normal CDF via erf (no scipy dependency)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over_normal(mean, sd, line):
    if sd <= 0 or np.isnan(sd) or np.isnan(mean) or np.isnan(line):
        return np.nan
    z = (line - mean) / sd
    return 1.0 - norm_cdf(z)


def kelly_fraction(p, odds_american, kelly_cap=0.10, kelly_multiplier=0.50):
    """
    Kelly sizing for a single bet.
    - kelly_multiplier: 0.5 means “half Kelly”
    - kelly_cap: max fraction of bankroll
    """
    try:
        p = float(p)
        if not (0 < p < 1):
            return 0.0
        dec = american_to_decimal(odds_american)
        b = dec - 1.0
        q = 1.0 - p
        f = (b * p - q) / b if b > 0 else 0.0
        f = max(0.0, f) * kelly_multiplier
        return float(min(f, kelly_cap))
    except Exception:
        return 0.0


def pretty_market_name(mkey: str) -> str:
    # Turn "player_pass_yds" -> "Player Pass Yds"
    return mkey.replace("_", " ").title()


def infer_position_from_market(market_key: str) -> str:
    mk = (market_key or "").lower()
    if "pass" in mk:
        return "QB"
    if "rush" in mk:
        return "RB"
    if "rec" in mk or "recept" in mk:
        return "WR/TE"
    if "anytime_td" in mk or "td" in mk:
        # could be RB/WR/TE; default mixed bucket
        return "SKILL"
    return "UNKNOWN"


def default_sd_for_position(pos: str, market_key: str) -> float:
    """
    Reasonable default SDs for yardage props by position bucket.
    (Used only if you don't upload SDs.)
    """
    pos = (pos or "").upper()
    mk = (market_key or "").lower()
    if "yd" not in mk and "yards" not in mk:
        return np.nan

    if pos == "QB":
        return 40.0
    if pos == "RB":
        return 25.0
    if pos in ("WR/TE", "WR", "TE"):
        return 30.0
    if pos == "SKILL":
        return 30.0
    return 30.0


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
    Normalizes /events/{event_id}/odds for ONE prop market into player-based rows.
    Outcome fields vary by book/market, but commonly:
      - outcome.description: player name
      - outcome.name: Over/Under (or sometimes player)
      - outcome.point: line
      - outcome.price: odds
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
                    or out.get("name")  # fallback
                    or ""
                ).strip()

                line = out.get("point")
                price = out.get("price")

                # If outcome.name is "Over/Under", player tends to be in description.
                # If outcome.name is player name, side may be missing; keep side blank.
                # We'll standardize later.
                rows.append(
                    dict(
                        Event=event_name,
                        Commence=commence,
                        Market=market_key,
                        Player=player,
                        Side=side,
                        Line=line,
                        Price=price,
                        Book=book,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)

    # Standardize Side: if not Over/Under/Yes/No, leave as-is
    df["Side"] = df["Side"].fillna("").astype(str)
    return df


def best_price_table(df: pd.DataFrame, group_cols: list[str], price_col: str = "Price") -> pd.DataFrame:
    """
    Best price across books:
    - For American odds, "best" is simply max numeric (e.g., -105 > -115, +200 > +180).
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
    """
    GET /sports/{sport_key}/markets
    Returns list of markets supported for that sport under your plan.
    """
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
    debug = {"url": url, "status": status, "ok": ok, "error": None, "sample": None}
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
    """
    GET /sports/{sport_key}/events/{event_id}/odds?markets={market_key}
    ONE market per request => avoids 422.
    """
    url = f"{BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": ODDS_FORMAT,
    }
    ok, payload, status = safe_get(url, params=params)
    debug = {"url": url, "status": status, "ok": ok, "error": None}
    if ok and isinstance(payload, dict):
        return normalize_event_props(payload, market_key), debug
    debug["error"] = payload
    return pd.DataFrame(), debug


# -----------------------------
# PROJECTIONS (REAL) — CSV + FALLBACKS
# -----------------------------
def parse_projection_csv(upload) -> pd.DataFrame:
    """
    Accepts CSV with columns (any subset is ok):
      player, market_key, mean, sd, position, p_yes
    Examples:
      Josh Allen, player_pass_yds, 258.4, 38, QB,
      Christian McCaffrey, player_anytime_td,,,,0.58
    """
    if upload is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(upload)
    except Exception:
        return pd.DataFrame()

    # standardize cols
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    if "player" in df.columns:
        rename["player"] = "Player"
    if "market_key" in df.columns:
        rename["market_key"] = "Market"
    if "mean" in df.columns:
        rename["mean"] = "Mean"
    if "sd" in df.columns:
        rename["sd"] = "SD"
    if "position" in df.columns:
        rename["position"] = "Pos"
    if "p_yes" in df.columns:
        rename["p_yes"] = "P_Yes"

    df = df.rename(columns=rename)
    for col in ("Player", "Market", "Pos"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    for col in ("Mean", "SD", "P_Yes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep only known cols
    keep = [c for c in ["Player", "Market", "Mean", "SD", "Pos", "P_Yes"] if c in df.columns]
    return df[keep].copy()


def build_player_model_prob(row, proj_df: pd.DataFrame) -> float:
    """
    Returns model probability for the offered selection.
    - Yardage props: assume Normal(mean, sd) and compute P(Over) or P(Under).
    - Anytime TD: uses P_Yes if available; otherwise uses conservative baseline from market.
    - If projections missing: returns np.nan; caller falls back to consensus implied.
    """
    market = row.get("Market", "")
    player = (row.get("Player", "") or "").strip()
    side = (row.get("Side", "") or "").strip().lower()
    line = row.get("Line", np.nan)

    # Try to match projection row by exact market+player, else by player only
    p = None
    if not proj_df.empty:
        m1 = proj_df[(proj_df.get("Market", "") == market) & (proj_df.get("Player", "") == player)]
        if m1.empty:
            m1 = proj_df[(proj_df.get("Player", "") == player)]
        if not m1.empty:
            p = m1.iloc[0].to_dict()
    if not p:
        return np.nan

    # Determine position bucket
    pos = (p.get("Pos") or infer_position_from_market(market)).upper()
    mean = p.get("Mean", np.nan)
    sd = p.get("SD", np.nan)

    # TD props often have no line
    if "anytime_td" in market or market.endswith("_td") or "td" in market:
        py = p.get("P_Yes", np.nan)
        if not np.isnan(py):
            # If selection is "Yes/No" sometimes, honor side if present
            if side in ("no", "under"):
                return 1.0 - float(py)
            return float(py)
        return np.nan

    # Yardage / volume
    if np.isnan(sd):
        sd = default_sd_for_position(pos, market)
    if np.isnan(mean) or np.isnan(sd) or np.isnan(line):
        return np.nan

    p_over = prob_over_normal(mean, sd, line)
    if side == "under":
        return 1.0 - p_over
    # default to over if not specified
    return p_over


def consensus_true_prob(df_group: pd.DataFrame) -> float:
    """
    Fallback “realistic” true-prob baseline when you don't have a projection:
    Use mean implied prob across books as consensus (does not remove vig perfectly,
    but is stable and avoids the old 50% slider).
    """
    vals = df_group["Implied"].dropna().values
    if len(vals) == 0:
        return np.nan
    # Clip slightly for stability
    return float(np.clip(vals.mean(), 0.02, 0.98))


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📊 Dashboard")

sport_name = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_name]

view = st.sidebar.radio("View", ["Game Lines", "Player Props"], index=0)

debug_mode = st.sidebar.toggle("Show Debug", value=False)

st.sidebar.markdown("---")
bankroll = st.sidebar.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
kelly_mult = st.sidebar.slider("Kelly Multiplier", 0.10, 1.00, 0.50, 0.05)  # half-kelly default
kelly_cap = st.sidebar.slider("Kelly Cap (max % bankroll)", 0.01, 0.25, 0.10, 0.01)
top_n = st.sidebar.slider("Auto-rank Top Bets", 2, 5, 3, 1)

st.sidebar.markdown("---")
proj_upload = st.sidebar.file_uploader(
    "Upload Projections CSV (optional)",
    type=["csv"],
    help="Columns: player, market_key, mean, sd, position, p_yes (TD).",
)

proj_df = parse_projection_csv(proj_upload)
if debug_mode:
    st.sidebar.markdown("**DEBUG: Projections rows**")
    st.sidebar.write(int(len(proj_df)))

# -----------------------------
# HEADER METRICS
# -----------------------------
c1, c2, c3 = st.columns([1, 1, 1])
c1.metric("Sport", sport_name)
c2.metric("Last Refresh", now_utc())
c3.metric("Projections Loaded", "Yes" if not proj_df.empty else "No")

tabs = st.tabs(["Dashboard", "All Books (Raw)", "Setup Help"])

# -----------------------------
# GAME LINES (SEPARATE CALL PATH)
# -----------------------------
if view == "Game Lines":
    chosen_human = st.sidebar.multiselect(
        "Game Line Markets",
        list(GAME_MARKETS.keys()),
        default=["Spreads", "Totals", "Moneyline (H2H)"],
    )
    chosen = [GAME_MARKETS[x] for x in chosen_human] or ["spreads"]

    df_lines, dbg_lines = api_fetch_game_lines(sport_key, chosen)
    if debug_mode:
        st.sidebar.markdown("**DEBUG (Game Lines Call)**")
        st.sidebar.write(dbg_lines)

    if df_lines.empty:
        st.warning("No game line data available for these settings.")
    else:
        # Best price per event/market/outcome/line
        raw = df_lines.copy()
        best = best_price_table(raw, ["Event", "Market", "Outcome", "Line"], price_col="Price")
        best["Best Implied"] = best["Best Price"].apply(american_to_implied)

        # “Real” probability baseline for game lines without a team model:
        # use consensus implied per (Event, Market, Outcome, Line) (across all books) as p_true.
        # Then EV becomes: p_true - implied(best).
        # This is conservative but stable until you add a team efficiency model.
        cons_map = (
            raw.groupby(["Event", "Market", "Outcome", "Line"])
            .apply(consensus_true_prob)
            .rename("P_true")
            .reset_index()
        )
        best = best.merge(cons_map, on=["Event", "Market", "Outcome", "Line"], how="left")
        best["EV"] = (best["P_true"] - best["Best Implied"]) * 100.0
        best["Kelly %"] = best.apply(
            lambda r: kelly_fraction(r["P_true"], r["Best Price"], kelly_cap=kelly_cap, kelly_multiplier=kelly_mult),
            axis=1,
        )
        best["Stake $"] = (best["Kelly %"] * bankroll).round(2)

        best = best.sort_values("EV", ascending=False)
        top = best.head(top_n).copy()

        with tabs[0]:
            st.subheader("🔥 Auto-Ranked Top Bets (Game Lines)")
            st.dataframe(
                top[
                    ["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV", "Kelly %", "Stake $"]
                ],
                use_container_width=True,
            )

            st.subheader("Best Price Board (Game Lines)")
            st.dataframe(
                best[
                    ["Event", "Market", "Outcome", "Line", "Best Price", "Best Book", "P_true", "Best Implied", "EV", "Kelly %", "Stake $"]
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
# PLAYER PROPS (SEPARATE CALL PATH + AUTO-DISCOVERY)
# -----------------------------
else:
    # 1) Discover available markets for this sport
    markets_list, dbg_mk = api_discover_markets(sport_key)
    if debug_mode:
        st.sidebar.markdown("**DEBUG (Market Discovery)**")
        st.sidebar.write(dbg_mk)

    # markets endpoint returns list of dicts like {"key": "...", "name": "..."} depending on plan
    discovered_keys = []
    discovered_map = {}
    for m in markets_list:
        if not isinstance(m, dict):
            continue
        k = m.get("key") or m.get("market_key") or ""
        n = m.get("name") or m.get("title") or pretty_market_name(k)
        if not k:
            continue
        discovered_keys.append(k)
        discovered_map[k] = n

    # Filter to player prop-ish keys (common pattern)
    # Keep it broad but avoid game line keys
    player_like = [
        k for k in discovered_keys
        if (k.startswith("player_") or "player" in k) and k not in ("player_props",)
    ]

    # Some plans name props without player_ prefix; keep a fallback group as well:
    # You can expand this list later if needed.
    additional_playerish = [k for k in discovered_keys if k.endswith("_player") or "anytime" in k]
    for k in additional_playerish:
        if k not in player_like:
            player_like.append(k)

    player_like = sorted(set(player_like))

    if not player_like:
        st.warning("No player prop markets were discovered for this sport on your plan/books right now.")
        st.stop()

    # 2) Populate dropdown dynamically
    def label_for(k):
        return discovered_map.get(k) or pretty_market_name(k)

    prop_market_key = st.sidebar.selectbox(
        "Player Prop Market (auto-discovered)",
        options=player_like,
        format_func=label_for,
        index=0,
    )

    # 3) Fetch events, then fetch props per event with ONE market per request (no 422)
    events, dbg_ev = api_fetch_events(sport_key)
    if debug_mode:
        st.sidebar.markdown("**DEBUG (Events Call)**")
        st.sidebar.write(dbg_ev)

    if not events:
        st.warning("No events returned for this sport right now.")
        st.stop()

    all_props = []
    failures = []

    with st.spinner("Loading player props (event-by-event)…"):
        # Cap fan-out to reduce quota usage; raise if your plan supports it.
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
                if debug_mode and dbg_p.get("ok") is False:
                    failures.append(dbg_p)
                continue

            all_props.append(df_evprops)

    if debug_mode and failures:
        st.sidebar.markdown("**DEBUG (Prop Failures / Skipped)**")
        st.sidebar.write(failures[:5])

    if not all_props:
        st.warning(
            "No player props came back for that market right now. "
            "Try a different prop market from the dropdown."
        )
        st.stop()

    raw_props = pd.concat(all_props, ignore_index=True)

    # Keep player-based rows only
    raw_props["Player"] = raw_props["Player"].fillna("").astype(str).str.strip()
    raw_props = raw_props[raw_props["Player"].str.len() > 0].copy()
    if raw_props.empty:
        st.warning("Props returned but player fields were empty. Turn on Debug to inspect raw payload.")
        st.stop()

    # Position bucket for modeling (market-driven unless CSV overrides)
    raw_props["PosBucket"] = raw_props["Market"].apply(infer_position_from_market)

    # Build model probabilities:
    # - Use projections CSV if available (Mean/SD or P_Yes)
    # - Else fallback to consensus implied (computed per player/side/line across books)
    raw_props["ModelProb"] = raw_props.apply(lambda r: build_player_model_prob(r, proj_df), axis=1)

    # Fallback: consensus implied per player/side/line
    group_cols_raw = ["Event", "Market", "Player", "Side", "Line"]
    cons = raw_props.groupby(group_cols_raw).apply(consensus_true_prob).rename("P_true").reset_index()

    # Best price per player/side/line across books
    best = best_price_table(raw_props, group_cols_raw, price_col="Price")
    best["Best Implied"] = best["Best Price"].apply(american_to_implied)

    # Merge consensus p_true
    best = best.merge(cons, on=group_cols_raw, how="left")

    # Merge projection-driven model prob if available:
    # We take the MAX ModelProb within the group (they should match anyway if projections are consistent)
    mp = raw_props.groupby(group_cols_raw)["ModelProb"].max().reset_index()
    best = best.merge(mp, on=group_cols_raw, how="left")

    # Choose “real projections” prob when present; else consensus implied baseline
    best["P_used"] = best["ModelProb"].where(~best["ModelProb"].isna(), best["P_true"])

    # EV and Kelly
    best["EV"] = (best["P_used"] - best["Best Implied"]) * 100.0
    best["Kelly %"] = best.apply(
        lambda r: kelly_fraction(r["P_used"], r["Best Price"], kelly_cap=kelly_cap, kelly_multiplier=kelly_mult),
        axis=1,
    )
    best["Stake $"] = (best["Kelly %"] * bankroll).round(2)

    best = best.sort_values("EV", ascending=False)
    top = best.head(top_n).copy()

    with tabs[0]:
        st.subheader(f"🔥 Auto-Ranked Top Bets (Player Props) — {label_for(prop_market_key)}")
        st.dataframe(
            top[
                ["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV", "Kelly %", "Stake $"]
            ],
            use_container_width=True,
        )

        st.subheader("Best Price Board (Player Props)")
        st.dataframe(
            best[
                ["Event", "Player", "Side", "Line", "Best Price", "Best Book", "PosBucket", "P_used", "Best Implied", "EV", "Kelly %", "Stake $"]
            ],
            use_container_width=True,
        )

    with tabs[1]:
        st.subheader("All Books (Raw Player Props)")
        st.dataframe(
            raw_props[["Event", "Market", "Player", "Side", "Line", "Price", "Book", "Implied", "PosBucket", "ModelProb"]],
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("Setup Help")
    st.markdown(
        """
**1) Why auto-discover markets?**  
Because prop market availability changes by **sport / week / book / plan**. This app pulls the live list from:

- `/v4/sports/{sport_key}/markets`

**2) Why event-by-event props? (Avoids 422 errors)**  
Props are fetched using:

- `/v4/sports/{sport_key}/events` (get event IDs)
- `/v4/sports/{sport_key}/events/{event_id}/odds?markets=<ONE_MARKET_KEY>` (one market per request)

**3) Projections CSV format (recommended for “real projections”)**  
Upload a CSV with columns like:

- `player` (required)
- `market_key` (recommended) e.g. `player_pass_yds`
- `mean` and `sd` for yardage props
- `position` (optional: QB/RB/WR/TE)
- `p_yes` for Anytime TD (probability the player scores)

Example:
```csv
player,market_key,mean,sd,position,p_yes
Josh Allen,player_pass_yds,258.4,38,QB,
Christian McCaffrey,player_anytime_td,,,,RB,0.58
