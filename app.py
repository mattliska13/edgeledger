import os
import time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, cleaner sidebar + nicer typography
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; }
      [data-testid="stSidebarNav"] { display: none; } /* hides multipage nav */
      [data-testid="stSidebar"] { padding-top: 0.8rem; }
      .big-title { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; }
      .subtle { color: rgba(250,250,250,0.75); }
      .pill { display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; background: rgba(255,255,255,0.08); margin-right: 0.4rem; }
      .ok { color: #7CFC00; font-weight: 700; }
      .warn { color: #FFD166; font-weight: 700; }
      .bad { color: #FF6B6B; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Keys (prefer Streamlit secrets / env vars) ---
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", None) if hasattr(st, "secrets") else None
DATAGOLF_KEY = st.secrets.get("DATAGOLF_KEY", None) if hasattr(st, "secrets") else None

ODDS_API_KEY = ODDS_API_KEY or os.getenv("ODDS_API_KEY") or "d1a096c07dfb711c63560fcc7495fd0d"
DATAGOLF_KEY = DATAGOLF_KEY or os.getenv("DATAGOLF_KEY") or "909304744927252dd7a207f7dce4"

BOOKMAKERS = "draftkings,fanduel"  # per your request
REGION = "us"
ODDS_HOST = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

# Player prop market keys you requested (Odds API keys)
PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_passing_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rushing_yds",
    "Receiving Yards": "player_receiving_yds",
    "Receptions": "player_receptions",
}

GAME_LINE_MARKETS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

# -----------------------------
# HELPERS
# -----------------------------
def now_et_str():
    # America/New_York is your timezone; keep it simple without pytz dependency
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def american_to_implied(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def safe_get(url, params, timeout=20):
    r = requests.get(url, params=params, timeout=timeout)
    # let caller see status + json; raise only in caller when desired
    return r

def is_list_of_dicts(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))

def debug_box(enabled: bool, title: str, payload):
    if not enabled:
        return
    with st.expander(f"🧪 Debug — {title}", expanded=False):
        st.write(payload)

def best_price_group(df: pd.DataFrame, group_cols: list, odds_col="Price"):
    """
    Choose the best price (max for American odds) within group.
    Returns a DF with best price + best book.
    """
    if df.empty:
        return df

    # For American odds, higher number is always better (e.g., +120 better than +110; -105 better than -110)
    df = df.copy()
    df["_PriceNum"] = pd.to_numeric(df[odds_col], errors="coerce")

    idx = df.groupby(group_cols)["_PriceNum"].idxmax()
    best = df.loc[idx].copy()

    best = best.rename(columns={
        "Book": "Best Book",
        odds_col: "Best Price"
    })

    best.drop(columns=["_PriceNum"], errors="ignore", inplace=True)
    return best

def compute_ev_from_market_avg(df: pd.DataFrame, group_cols: list):
    """
    EV uses:
    - "model_prob" = no external projection here -> we approximate fair prob from market average (viggy)
      then adjust with lightweight heuristics (optional).
    - implied prob from the BEST PRICE (actual bet)
    EV = best_price_prob - implied_prob (scaled for sorting)
    """
    if df.empty:
        return df

    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)

    # Market average implied prob within same selection (across books)
    avg = df.groupby(group_cols)["Implied"].mean().reset_index().rename(columns={"Implied": "MarketProb"})
    df = df.merge(avg, on=group_cols, how="left")

    # Use MarketProb as baseline "model_prob"
    df["ModelProb"] = df["MarketProb"].clip(0.001, 0.999)

    # Simple EV proxy: difference in prob * 100 (sortable). (You can replace with your projections later.)
    df["EV"] = (df["ModelProb"] - df["Implied"]) * 100.0
    return df

def enforce_no_contradictions(df_best: pd.DataFrame, contradiction_key_cols: list, side_col="Side"):
    """
    Prevent contradictory picks by selecting ONLY the highest EV per contradiction key.
    Example totals: same Event + Market + Line -> keep only best between Over/Under.
    Example spreads: same Event + Market -> keep only best between teams for same line.
    """
    if df_best.empty:
        return df_best
    df_best = df_best.copy()
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").fillna(-1e9)
    keep_idx = df_best.groupby(contradiction_key_cols)["EV"].idxmax()
    return df_best.loc[keep_idx].sort_values("EV", ascending=False)

# -----------------------------
# ODDS API — GAME LINES
# -----------------------------
@st.cache_data(ttl=60*60)  # 1 hour cache (keeps usage down)
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
    return r.status_code, r.json() if "application/json" in r.headers.get("Content-Type","") else r.text

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
                    rows.append({
                        "Event": event,
                        "Commence": commence,
                        "Market": mkey,
                        "Side": out.get("name"),
                        "Line": out.get("point"),
                        "Price": out.get("price"),
                        "Book": book,
                    })
    df = pd.DataFrame(rows)
    # drop rows missing required fields
    if df.empty:
        return df
    df = df.dropna(subset=["Market", "Side", "Price"])
    return df

# -----------------------------
# ODDS API — PLAYER PROPS (2-step: events -> event odds)
# -----------------------------
@st.cache_data(ttl=60*60)  # 1 hour cache
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    r = safe_get(url, params=params)
    return r.status_code, r.json() if "application/json" in r.headers.get("Content-Type","") else r.text

@st.cache_data(ttl=60*60)  # 1 hour cache
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
    payload = r.json() if "application/json" in r.headers.get("Content-Type","") else r.text
    return r.status_code, payload

def normalize_player_props(event_payload):
    """
    Event-odds payload shape includes:
      event info + bookmakers -> markets -> outcomes
    Outcomes often include:
      - name (player)
      - description (Over/Under or TD etc; varies)
      - point (line)
      - price (odds)
    We'll build:
      Event, Market, Player, Side, Line, Price, Book
    """
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
                player = out.get("name")  # usually player
                side = out.get("description") or out.get("name")  # over/under or player label
                line = out.get("point")
                price = out.get("price")
                # If description is missing, try to infer side for OU style props
                # (Keep as-is; we mainly avoid contradictions by grouping)
                rows.append({
                    "Event": event,
                    "Market": mkey,
                    "Player": player,
                    "Side": side,
                    "Line": line,
                    "Price": price,
                    "Book": book,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["Market", "Player", "Price"])
    return df

# -----------------------------
# DATAGOLF — PGA (Win + Top10)
# -----------------------------
DG_BASE = "https://feeds.datagolf.com"

@st.cache_data(ttl=60*60)  # 1 hour cache
def datagolf_pre_tournament(tour="pga"):
    url = f"{DG_BASE}/preds/pre-tournament"
    params = {
        "tour": tour,
        "add_position": "10",   # ensure Top-10 is present
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_KEY,
    }
    r = safe_get(url, params=params)
    return r.status_code, r.json() if "application/json" in r.headers.get("Content-Type","") else r.text

@st.cache_data(ttl=60*60)
def datagolf_skill_ratings():
    # Skill ratings: returns per-skill estimates (often includes putting/approach/etc).
    url = f"{DG_BASE}/preds/skill-ratings"
    params = {
        "display": "value",
        "file_format": "json",
        "key": DATAGOLF_KEY,
    }
    r = safe_get(url, params=params)
    return r.status_code, r.json() if "application/json" in r.headers.get("Content-Type","") else r.text

def normalize_datagolf(pred_payload, skill_payload=None):
    """
    Pre-tournament payload contains baseline + baseline+course models.
    We'll use the model that includes course history & fit when available.
    We ALSO incorporate skill-ratings if we can find putting / tee-to-green-ish fields.
    """
    if not isinstance(pred_payload, dict):
        return pd.DataFrame()

    # Find a list-like field with player rows
    # DataGolf schema can vary; try common keys
    candidates = []
    for k, v in pred_payload.items():
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            candidates.append(k)

    if not candidates:
        return pd.DataFrame()

    # pick the biggest list as "main table"
    main_key = sorted(candidates, key=lambda x: len(pred_payload.get(x, [])), reverse=True)[0]
    rows = pred_payload.get(main_key, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Try to locate player name and probabilities
    # Common columns include: player_name, win, top_10, win_baseline, win_course, etc.
    # We'll attempt to use "course history & fit" variant if present.
    name_col = "player_name" if "player_name" in df.columns else ("player" if "player" in df.columns else None)
    if not name_col:
        # fallback: first stringish column
        for c in df.columns:
            if df[c].dtype == "object":
                name_col = c
                break

    df = df.rename(columns={name_col: "Player"}) if name_col else df
    if "Player" not in df.columns:
        df["Player"] = "Unknown"

    # Probability columns (percent)
    # Prefer course-history/fit model columns if present
    # We’ll try a few common patterns, else fallback to any "win"/"top_10"
    def pick_prob_col(prefix_candidates):
        for c in df.columns:
            lc = c.lower()
            if any(p in lc for p in prefix_candidates):
                return c
        return None

    win_col = None
    top10_col = None

    # likely names
    for c in df.columns:
        lc = c.lower()
        if "win" == lc or lc.endswith("_win") or lc.startswith("win"):
            win_col = win_col or c
        if "top_10" in lc or "top10" in lc or lc.endswith("_10") or "top_10" == lc:
            top10_col = top10_col or c

    # If there's a course-history/fit variant, try to prefer it
    # This is a heuristic: any column containing "course" or "fit" and "win/top_10"
    for c in df.columns:
        lc = c.lower()
        if ("course" in lc or "fit" in lc) and "win" in lc:
            win_col = c
        if ("course" in lc or "fit" in lc) and ("top_10" in lc or "top10" in lc):
            top10_col = c

    if not win_col:
        win_col = pick_prob_col(["win"])
    if not top10_col:
        top10_col = pick_prob_col(["top_10", "top10"])

    if not win_col or not top10_col:
        # Still show whatever we can, but scoring will be limited
        df["Win%"] = np.nan
        df["Top10%"] = np.nan
    else:
        df["Win%"] = pd.to_numeric(df[win_col], errors="coerce")
        df["Top10%"] = pd.to_numeric(df[top10_col], errors="coerce")

    # --- Skill ratings merge (tee-to-green + putting) ---
    sg_t2g = np.nan
    sg_putt = np.nan

    if isinstance(skill_payload, dict):
        # Again, schema varies; find a list table
        skill_list = None
        for k, v in skill_payload.items():
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
                skill_list = v
                break
        if skill_list:
            sk = pd.DataFrame(skill_list)
            # find name col
            sk_name = "player_name" if "player_name" in sk.columns else ("player" if "player" in sk.columns else None)
            if sk_name:
                sk = sk.rename(columns={sk_name: "Player"})
                # try to detect t2g + putting columns
                # tee-to-green often combines approach + off-the-tee + around-green
                # If a direct "sg_t2g" exists, use it; else approximate if approach/ott/arg exist.
                cols = {c.lower(): c for c in sk.columns}

                putt_col = cols.get("putting") or cols.get("sg_putt") or cols.get("sg_putting")
                ott_col = cols.get("ott") or cols.get("off_the_tee") or cols.get("sg_ott")
                app_col = cols.get("approach") or cols.get("sg_app") or cols.get("app")
                arg_col = cols.get("around_green") or cols.get("sg_arg") or cols.get("arg")
                t2g_col = cols.get("t2g") or cols.get("sg_t2g")

                if putt_col:
                    sk["SG_Putt"] = pd.to_numeric(sk[putt_col], errors="coerce")
                else:
                    sk["SG_Putt"] = np.nan

                if t2g_col:
                    sk["SG_T2G"] = pd.to_numeric(sk[t2g_col], errors="coerce")
                else:
                    parts = []
                    for cc in [ott_col, app_col, arg_col]:
                        if cc:
                            parts.append(pd.to_numeric(sk[cc], errors="coerce"))
                    sk["SG_T2G"] = np.nan if not parts else np.nanmean(np.vstack([p.to_numpy() for p in parts]), axis=0)

                sk2 = sk[["Player", "SG_T2G", "SG_Putt"]].drop_duplicates("Player")
                df = df.merge(sk2, on="Player", how="left")

    # --- Score using: course history & fit (baked into Win%/Top10% if those columns were course-based),
    # plus tee-to-green + putting
    # Normalize skill components to z-scores so they can blend with probabilities
    df["SG_T2G_z"] = (df["SG_T2G"] - df["SG_T2G"].mean()) / (df["SG_T2G"].std() + 1e-9)
    df["SG_Putt_z"] = (df["SG_Putt"] - df["SG_Putt"].mean()) / (df["SG_Putt"].std() + 1e-9)

    # Final score emphasizes "chance to happen" per your request:
    # Win picks -> mostly Win%, supported by T2G + Putt
    df["Score_Win"] = (df["Win%"].fillna(0) * 1.0) + (df["SG_T2G_z"].fillna(0) * 0.6) + (df["SG_Putt_z"].fillna(0) * 0.35)
    df["Score_Top10"] = (df["Top10%"].fillna(0) * 1.0) + (df["SG_T2G_z"].fillna(0) * 0.45) + (df["SG_Putt_z"].fillna(0) * 0.25)

    # Keep columns
    keep = ["Player", "Win%", "Top10%", "SG_T2G", "SG_Putt", "Score_Win", "Score_Top10"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    return df

# -----------------------------
# SIDEBAR (clean)
# -----------------------------
st.sidebar.markdown('<div class="big-title">Dashboard</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="subtle">Game lines • Player props • PGA</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

section = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA"], index=0)

sport = None
if section in ["Game Lines", "Player Props"]:
    sport = st.sidebar.selectbox("Sport", list(SPORT_KEYS.keys()), index=0)

debug = st.sidebar.checkbox("Show debug logs", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="pill">Books: DraftKings + FanDuel</span>', unsafe_allow_html=True)
st.sidebar.markdown(f'<span class="pill">Updated: {now_et_str()}</span>', unsafe_allow_html=True)

# -----------------------------
# MAIN TITLE
# -----------------------------
st.markdown('<div class="big-title">EdgeLedger</div>', unsafe_allow_html=True)
st.caption("Best Price • EV ranking • No-contradiction picks • Separate calls for game lines vs props")

# -----------------------------
# GAME LINES VIEW
# -----------------------------
if section == "Game Lines":
    sport_key = SPORT_KEYS[sport]

    market_label = st.selectbox("Bet Type", list(GAME_LINE_MARKETS.keys()), index=1)
    market_key = GAME_LINE_MARKETS[market_label]

    status, raw = fetch_game_lines(sport_key)
    debug_box(debug, f"Odds API /odds status={status}", raw if debug else {"status": status})

    if status != 200 or not is_list_of_dicts(raw):
        st.error("No game line data returned from the API (check key/plan/limits).")
        st.stop()

    df = normalize_game_lines(raw)
    if df.empty:
        st.warning("No game line rows were normalized from the API response.")
        st.stop()

    # Filter to selected market
    df = df[df["Market"] == market_key].copy()
    if df.empty:
        st.warning("No rows for this market right now.")
        st.stop()

    # Grouping keys for market-average prob and best-price selection
    # For totals: group by Event + Market + Side + Line
    # For spreads: group by Event + Market + Side + Line
    # For h2h: group by Event + Market + Side (no line)
    group_cols = ["Event", "Market", "Side"] + (["Line"] if market_key in ["totals", "spreads"] else [])

    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)
    # Choose best price per selection
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")

    # Recompute implied + EV based on best price vs market-average modelprob
    # (We stored modelprob in df_ev, but df_best is subset. Merge MarketProb/ModelProb from df_ev.)
    cols_to_merge = group_cols + ["MarketProb", "ModelProb"]
    df_best = df_best.merge(
        df_ev[cols_to_merge].drop_duplicates(group_cols),
        on=group_cols,
        how="left",
    )
    df_best["Implied"] = pd.to_numeric(df_best["Best Price"], errors="coerce").apply(american_to_implied)
    df_best["EV"] = (df_best["ModelProb"] - df_best["Implied"]) * 100.0

    # --- No-contradictions rule ---
    # Totals: contradiction key = Event + Market + Line  (Over vs Under)
    # Spreads: contradiction key = Event + Market        (team A vs team B)
    # H2H:     contradiction key = Event + Market        (team A vs team B)
    if market_key == "totals":
        df_best = enforce_no_contradictions(df_best, ["Event", "Market", "Line"])
    else:
        df_best = enforce_no_contradictions(df_best, ["Event", "Market"])

    # Auto-ranked top 2–5 (dynamic, but within bounds)
    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top)

    st.subheader(f"{sport} — {market_label}")
    st.markdown("### Top Bets Ranked by EV")
    show_cols = ["Event", "Side"]
    if "Line" in df_best.columns:
        show_cols += ["Line"]
    show_cols += ["Best Price", "Best Book", "ModelProb", "Implied", "EV"]
    show_cols = [c for c in show_cols if c in df_best.columns]

    # add a visual highlight column (no Styler issues)
    top_bets = top_bets.copy()
    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.dataframe(
        top_bets[["Event", "Side"] + (["Line"] if "Line" in top_bets.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]],
        use_container_width=True,
    )

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)
    st.dataframe(
        snap[["Event", "Side"] + (["Line"] if "Line" in snap.columns else []) + ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]],
        use_container_width=True,
    )

# -----------------------------
# PLAYER PROPS VIEW
# -----------------------------
elif section == "Player Props":
    sport_key = SPORT_KEYS[sport]

    prop_label = st.selectbox("Prop Type", list(PLAYER_PROP_MARKETS.keys()), index=0)
    prop_market = PLAYER_PROP_MARKETS[prop_label]

    # 1) events
    e_status, events_raw = fetch_events(sport_key)
    debug_box(debug, f"Odds API /events status={e_status}", events_raw if debug else {"status": e_status})

    if e_status != 200 or not is_list_of_dicts(events_raw):
        st.error("No events returned from the API. Check plan/limits or sport_key.")
        st.stop()

    event_ids = []
    for ev in events_raw:
        # Only upcoming (basic filter: require id + teams)
        if ev.get("id") and ev.get("home_team") and ev.get("away_team"):
            event_ids.append(ev["id"])

    # If you want fewer calls, limit scan count
    max_events_to_scan = st.sidebar.slider("Events to scan (usage control)", 1, 10, 5)
    event_ids = event_ids[:max_events_to_scan]

    if not event_ids:
        st.warning("No upcoming event IDs found to query for props.")
        st.stop()

    # 2) per-event odds for this prop market
    all_props = []
    debug_rows = []

    for eid in event_ids:
        p_status, payload = fetch_event_props(sport_key, eid, prop_market)
        debug_rows.append({
            "event_id": eid,
            "market": prop_market,
            "status": p_status,
            "ok": (p_status == 200),
        })

        # Skip unsupported markets/events (422) without killing the page
        if p_status != 200 or not isinstance(payload, dict):
            continue

        dfp = normalize_player_props(payload)
        if not dfp.empty:
            all_props.append(dfp)

        # small sleep to be polite to API
        time.sleep(0.15)

    debug_box(debug, "Props per-event call results", debug_rows)

    if not all_props:
        st.warning(
            "No player props were returned for this prop type from DraftKings/FanDuel for the scanned events.\n\n"
            "This usually means **the market isn’t offered** for those events/books right now (API returns 422 per event)."
        )
        st.stop()

    df = pd.concat(all_props, ignore_index=True)

    # IMPORTANT: Ensure we're truly player-based (not team)
    # Some markets can return team-like outcomes; keep rows where Player isn't one of the teams or isn't 'Over/Under'
    df["Player"] = df["Player"].astype(str)
    df = df[~df["Player"].str.lower().isin(["over", "under"])]
    df = df.dropna(subset=["Price"])

    # Compute EV vs market-average implied prob per selection
    # Group by Event + Market + Player + Side + Line
    group_cols = ["Event", "Market", "Player", "Side"] + (["Line"] if "Line" in df.columns else [])
    df_ev = compute_ev_from_market_avg(df, group_cols=group_cols)

    # Best price per selection
    df_best = best_price_group(df_ev, group_cols=group_cols, odds_col="Price")

    # Merge MarketProb/ModelProb from df_ev
    cols_to_merge = group_cols + ["MarketProb", "ModelProb"]
    df_best = df_best.merge(
        df_ev[cols_to_merge].drop_duplicates(group_cols),
        on=group_cols,
        how="left",
    )
    df_best["Implied"] = pd.to_numeric(df_best["Best Price"], errors="coerce").apply(american_to_implied)
    df_best["EV"] = (df_best["ModelProb"] - df_best["Implied"]) * 100.0

    # --- No-contradictions for props ---
    # For OU props: same Event+Market+Player+Line can have Over/Under (or equivalent)
    # Keep only best EV among contradictory sides.
    contradiction_cols = ["Event", "Market", "Player"]
    if "Line" in df_best.columns:
        contradiction_cols += ["Line"]
    df_best = enforce_no_contradictions(df_best, contradiction_cols)

    # Auto-ranked top 2–5
    n_top = int(np.clip(len(df_best), 2, 5))
    top_bets = df_best.sort_values("EV", ascending=False).head(n_top).copy()

    st.subheader(f"{sport} — Player Props")
    st.caption(f"Prop Type: {prop_label} • Scanned events: {len(event_ids)} • Normalized props: {len(df_best)}")

    top_bets["⭐ Best Book"] = "⭐ " + top_bets["Best Book"].astype(str)

    st.markdown("### Top Bets Ranked by EV (No Contradictions)")
    cols = ["Event", "Player", "Side"]
    if "Line" in top_bets.columns:
        cols += ["Line"]
    cols += ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(top_bets[cols], use_container_width=True)

    st.markdown("### Snapshot — Top 25 (by EV)")
    snap = df_best.sort_values("EV", ascending=False).head(25).copy()
    snap["⭐ Best Book"] = "⭐ " + snap["Best Book"].astype(str)

    cols2 = ["Event", "Player", "Side"]
    if "Line" in snap.columns:
        cols2 += ["Line"]
    cols2 += ["Best Price", "⭐ Best Book", "ModelProb", "Implied", "EV"]
    st.dataframe(snap[cols2], use_container_width=True)

# -----------------------------
# PGA VIEW
# -----------------------------
else:
    st.subheader("PGA — Picks (Win + Top-10)")
    st.caption("Uses DataGolf pre-tournament probabilities + skill ratings (tee-to-green + putting) when available.")

    p_status, preds = datagolf_pre_tournament(tour="pga")
    s_status, skills = datagolf_skill_ratings()

    debug_box(debug, f"DataGolf pre-tournament status={p_status}", preds if debug else {"status": p_status})
    debug_box(debug, f"DataGolf skill-ratings status={s_status}", skills if debug else {"status": s_status})

    if p_status != 200 or not isinstance(preds, dict):
        st.error("No DataGolf predictions available (check key or membership/access).")
        st.stop()

    skill_payload = skills if (s_status == 200 and isinstance(skills, dict)) else None
    df = normalize_datagolf(preds, skill_payload=skill_payload)

    if df.empty:
        st.warning("No PGA rows could be normalized from the DataGolf response.")
        st.stop()

    # Top 5 picks each (Win + Top10) — and not contradictory since they’re separate markets by definition
    win_top5 = df.sort_values("Score_Win", ascending=False).head(5).copy()
    top10_top5 = df.sort_values("Score_Top10", ascending=False).head(5).copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🏆 Top 5 — Win")
        st.dataframe(win_top5[["Player", "Win%", "SG_T2G", "SG_Putt", "Score_Win"]], use_container_width=True)
        st.caption("Logic uses Win% (course fit/history model where available) + tee-to-green + putting.")

    with c2:
        st.markdown("### 🎯 Top 5 — Top-10")
        st.dataframe(top10_top5[["Player", "Top10%", "SG_T2G", "SG_Putt", "Score_Top10"]], use_container_width=True)
        st.caption("Logic uses Top10% + tee-to-green + putting.")

    st.markdown("### Snapshot — Top 25 (Win Score)")
    st.dataframe(df.sort_values("Score_Win", ascending=False).head(25), use_container_width=True)
