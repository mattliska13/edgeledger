def render_tracker():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Tracker — Pick Rate + Hit Rate")
    st.caption("Log picks from any section. Use ESPN auto-fill to grade Game Lines (Moneyline/Spreads/Totals) when finals are available.")

    df = _load_tracker()

    # -------------------------------
    # ESPN AUTO-GRADE CONTROLS
    # -------------------------------
    st.markdown("### ✅ Auto-fill Results (ESPN)")
    st.caption("Auto-grades **Game Lines** only (Moneyline/Spreads/Totals) for NFL/CFB/CBB when games are FINAL on ESPN.")

    colA, colB, colC = st.columns([1.3, 1.0, 1.0])
    with colA:
        auto_window = st.selectbox("Which tracker window to update?", ["Today", "This Week", "This Month", "This Year"], index=0)
    with colB:
        only_pending = st.checkbox("Only pending rows", value=True)
    with colC:
        run_auto = st.button("Auto-fill from ESPN")

    if run_auto:
        df_work = df.copy()

        # window filter based on LoggedAt
        win_map = _windows(df_work)
        sub = win_map.get(auto_window, df_work)

        # indices to attempt
        idxs = sub.index.tolist()

        # filter for game lines rows + supported sports
        def is_game_lines_row(r):
            return str(r.get("Mode", "")).strip() == "Game Lines"

        supported = {"NFL", "CFB", "CBB"}

        updates = 0
        checked = 0

        # build list of unique (sport, date) we need to pull
        need = set()
        for i in idxs:
            r = df_work.loc[i]
            if not is_game_lines_row(r):
                continue
            if only_pending and str(r.get("Status", "")).strip() == "Graded":
                continue
            sport = str(r.get("Sport", "")).strip()
            if sport not in supported:
                continue

            # Use LoggedAt date (local) to choose scoreboard date.
            dt = pd.to_datetime(r.get("LoggedAt", ""), errors="coerce")
            if pd.isna(dt):
                continue
            need.add((sport, dt.strftime("%Y%m%d")))

        finals_cache = {}
        for sport, yyyymmdd in sorted(list(need)):
            finals_map, meta = _espn_build_finals_map(sport, yyyymmdd)
            finals_cache[(sport, yyyymmdd)] = finals_map

        # apply grading
        for i in idxs:
            r = df_work.loc[i]
            if not is_game_lines_row(r):
                continue
            if only_pending and str(r.get("Status", "")).strip() == "Graded":
                continue

            sport = str(r.get("Sport", "")).strip()
            if sport not in supported:
                continue

            dt = pd.to_datetime(r.get("LoggedAt", ""), errors="coerce")
            if pd.isna(dt):
                continue
            yyyymmdd = dt.strftime("%Y%m%d")

            finals_map = finals_cache.get((sport, yyyymmdd), {})
            final = _match_final_for_tracker_event(finals_map, str(r.get("Event","")))

            checked += 1
            res = _grade_game_line_row(r, final)
            if res in ["W", "L", "P"]:
                df_work.at[i, "Result"] = res
                df_work.at[i, "Status"] = "Graded"
                updates += 1

        _save_tracker(df_work)
        df = df_work

        st.success(f"Auto-fill complete ✅ Checked: {checked} | Updated (Graded): {updates}")

    st.markdown("---")

    # -------------------------------
    # Summary
    # -------------------------------
    win_map = _windows(df)
    tables = []
    for label, sub in win_map.items():
        s = _summary_pick_rate(sub, label)
        if not s.empty:
            tables.append(s)

    if tables:
        summary = pd.concat(tables, ignore_index=True)
        st.markdown("### Summary (Today / Week / Month / Year)")
        st.dataframe(
            summary[["Window","Mode","Picks","Graded","Wins","Losses","Pushes","HitRate%"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No tracked picks yet. Log picks from Game Lines / Props / PGA.")

    # -------------------------------
    # Manual Grade Editor (still available)
    # -------------------------------
    st.markdown("### Grade Picks (Manual override)")
    st.caption("You can still manually edit results. ESPN auto-fill only affects Game Lines that are FINAL.")

    if df.empty:
        st.info("Tracker is empty.")
    else:
        df["Status"] = df["Status"].fillna("Pending")
        df["Result"] = df["Result"].fillna("")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="tracker_editor")
        if st.button("Save Tracker"):
            _save_tracker(edited)
            st.success("Saved.")

    st.markdown("</div>", unsafe_allow_html=True)
