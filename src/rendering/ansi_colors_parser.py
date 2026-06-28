import re

ANSI_256_COLORS_PATTERN = re.compile(r"\x1b\[(\d+);5;(\d+)m")
ANSI_CODE_PATTERN = re.compile(r"\x1b\[.*?m")

ANSI_256_FG_COLOR_ID = 38
ANSI_256_BG_COLOR_ID = 48

ANSI_16_COLORS_TO_RGB = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255)
]


def ansi_256_id_to_rgb(id):
    if id < 16:
        return ANSI_16_COLORS_TO_RGB[id]
    if id > 231:
        id -= 232
        gray = int(id / 23 * 255)
        return (gray, gray, gray)

    id -= 16
    r = id // 36
    r = int(r / 5 * 255)
    id %= 36
    g = id // 6
    g = int(g / 5 * 255)
    b = id % 6
    b = int(b / 5 * 255)
    return (r, g, b)


def strip_ansi_codes(symbol):
    return re.sub(ANSI_CODE_PATTERN, '', symbol)


def parse_ansi_colors(symbol):
    m = ANSI_256_COLORS_PATTERN.findall(symbol)
    chars = strip_ansi_codes(symbol)
    bg_color = None
    fg_color = None
    for parsed_color_id in m:
        if (int(parsed_color_id[0]) == ANSI_256_FG_COLOR_ID):
            fg_color = ansi_256_id_to_rgb(int(parsed_color_id[1]))
        elif (int(parsed_color_id[0]) == ANSI_256_BG_COLOR_ID):
            bg_color = ansi_256_id_to_rgb(int(parsed_color_id[1]))

    return {
        "chars": chars,
        "bg_color": bg_color,
        "fg_color": fg_color
    }
