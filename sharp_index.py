
SHARP_BOOKS = {"pinnacle": 1.4, "circa": 1.35, "betcris": 1.25}

def sharp_signal(open_line, current_line, book):
    weight = SHARP_BOOKS.get(book, 0)
    move = abs(current_line - open_line)
    return round(move * weight, 2)
