else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")

    if not DATAGOLF_API_KEY.strip():
        st.warning('DATAGOLF key not set. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY="...").')
        st.stop()

    # Controls
    if compact:
        only_value = st.toggle("Show +EV only (requires book odds)", value=False, key="pga_only_ev_m")
        top_n = st.slider("Top picks", 5, 20, 10, key="pga_top_m")
    else:
        c1, c2 = st.columns([1, 1])
        only_value = c1.toggle("Show +EV only (requires book odds)", value=False, key="pga_only_ev")
        top_n = c2.slider("Top picks", 5, 20, 10, key="pga_top")

    pre = dg_pre_tournament(tour="pga", add_position=10)
    decomp = dg_decompositions(tour="pga")
    skill = dg_skill_ratings(tour="pga")

    if not pre["ok"]:
        st.error("DataGolf pre-tournament feed failed.")
        st.stop()

    # Read meta + models
    meta = pre["payload"] if isinstance(pre["payload"], dict) else {}
    models_available = meta.get("models_available", [])
    if not isinstance(models_available, list) or not models_available:
        # fallback if meta missing for some reason
        models_available = ["baseline", "baseline_history_fit"]

    # Choose model (this is the missing piece causing 'parsed none')
    model_key = st.selectbox(
        "Model",
        options=models_available,
        index=0 if "baseline_history_fit" not in models_available else models_available.index("baseline_history_fit"),
        help="DataGolf returns predictions under these model keys. baseline_history_fit usually includes course history/fit."
    )

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]},
            "dg_decomp": {"ok": decomp["ok"], "status": decomp["status"], "url": decomp["url"]},
            "dg_skill": {"ok": skill["ok"], "status": skill["status"], "url": skill["url"]},
            "dg_meta": {
                "event_name": meta.get("event_name"),
                "last_updated": meta.get("last_updated"),
                "models_available": meta.get("models_available"),
                "using_model": model_key
            }
        })

    dfpga = normalize_pga(
        pre["payload"],
        model_key=model_key,
        decomp_payload=(decomp["payload"] if decomp["ok"] else None),
        skill_payload=(skill["payload"] if skill["ok"] else None),
    )

    if dfpga.empty:
        st.error(f"No PGA prediction rows returned from DataGolf for model='{model_key}'.")
        st.stop()

    # WIN board (edge if odds exist; else prob+analytics)
    win = dfpga.copy()
    win["SortKey"] = np.where(
        win["WinEdge"].notna(),
        win["WinEdge"],
        win["WinProb"].fillna(0) + 0.02 * win["AnalyticsScore"].fillna(0)
    )
    win = win.sort_values("SortKey", ascending=False)

    if only_value:
        win = win[win["WinEdge"].fillna(-1) > 0]

    win_top = win.head(int(top_n)).copy()
    win_cols = ["Player", "WinProb%", "BestWinOddsDisp", "BestWinBook", "WinEdge%"]
    for extra in ["sg_t2g", "sg_putt", "bogey_avoidance", "AnalyticsScore"]:
        if extra in win_top.columns:
            win_cols.append(extra)
    win_cols = [c for c in win_cols if c in win_top.columns]

    st.markdown("### Top Win Candidates")
    st.dataframe(win_top[win_cols], use_container_width=True, hide_index=True)

    # TOP-10 board
    t10 = dfpga.copy()
    t10["SortKey"] = t10["Top10Prob"].fillna(0) + 0.03 * t10["AnalyticsScore"].fillna(0)
    t10 = t10.sort_values("SortKey", ascending=False)
    t10_top = t10.head(int(top_n)).copy()

    t10_cols = ["Player", "Top10Prob%"]
    for extra in ["sg_t2g", "sg_putt", "bogey_avoidance", "AnalyticsScore"]:
        if extra in t10_top.columns:
            t10_cols.append(extra)
    t10_cols = [c for c in t10_cols if c in t10_top.columns]

    st.markdown("### Top-10 Candidates")
    st.dataframe(t10_top[t10_cols], use_container_width=True, hide_index=True)

    # Charts
    st.markdown("#### Probability view (Top picks)")
    cwin = win_top.copy()
    cwin["Label"] = cwin["Player"].astype(str)
    cwin["WinProbPct"] = cwin["WinProb%"]
    bar_prob(cwin, "Label", "WinProbPct", "Win Probability (Top Picks)")

    ct10 = t10_top.copy()
    ct10["Label"] = ct10["Player"].astype(str)
    ct10["Top10ProbPct"] = ct10["Top10Prob%"]
    bar_prob(ct10, "Label", "Top10ProbPct", "Top-10 Probability (Top Picks)")

    st.markdown("</div>", unsafe_allow_html=True)
