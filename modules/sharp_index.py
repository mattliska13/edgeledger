def sharp_indicator(open_line, current_line, public_pct):
    """
    Sharp signal increases when:
    - Line moves against public
    - Public > 60% but line doesn't follow
    """
    move = current_line - open_line
    if public_pct > 0.6 and move <= 0:
        return "SHARP"
    if public_pct < 0.4 and move >= 0:
        return "SHARP"
    return "PUBLIC"
