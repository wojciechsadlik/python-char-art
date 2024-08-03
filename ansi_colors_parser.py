import re
from ansi_colorizer import ansi_256_id_to_rgb

ANSI_256_COLORS_PATTERN = re.compile(r"\x1b\[(\d+);5;(\d+)m")
ANSI_CODE_PATTERN = re.compile(r"\x1b\[.*?m")

ANSI_256_FG_COLOR_ID = 38
ANSI_256_BG_COLOR_ID = 48


def strip_ansi_codes(chars):
    return re.sub(ANSI_CODE_PATTERN, '', chars)


def parse_ansi_colors(chars):
    m = ANSI_256_COLORS_PATTERN.findall(chars)
    char = strip_ansi_codes(chars)
    bg_color = None
    fg_color = None
    for parsed_color_id in m:
        if (int(parsed_color_id[0]) == ANSI_256_FG_COLOR_ID):
            fg_color = ansi_256_id_to_rgb(int(parsed_color_id[1]))
        elif (int(parsed_color_id[0]) == ANSI_256_BG_COLOR_ID):
            bg_color = ansi_256_id_to_rgb(int(parsed_color_id[1]))

    return {
        "char": char,
        "bg_color": bg_color,
        "fg_color": fg_color
    }
