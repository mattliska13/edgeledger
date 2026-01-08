def sharp_index(line_open, line_current, handle_pct):
    """
    Simple market-maker detection proxy
    """
    move = abs(line_current - line_open)
    return round(move * (1 - handle_pct), 3)
