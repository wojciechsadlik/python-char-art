import numpy as np

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


def set_char_fg_256_color_code(id):
    return f'\x1b[38;5;{id}m'


def set_char_bg_256_color_code(id):
    return f'\x1b[48;5;{id}m'


def set_char_fg_rgb_color_code(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return f'\x1b[38;2;{r};{g};{b}m'


def set_char_bg_rgb_color_code(r, g, b):
    r, g, b = int(r), int(g), int(b)
    return f'\x1b[48;2;{r};{g};{b}m'


def reset_code():
    return '\x1b[0m'


def rgb_to_ansi_256_id(r, g, b):
    r = int(r / 255 * 5)
    g = int(g / 255 * 5)
    b = int(b / 255 * 5)
    return 16 + 36 * r + 6 * g + 1 * b


def scale_pix_rgb_brightness(pix_rgb, scale):
    if (scale <= 1):
        return scale * pix_rgb

    scale -= 1
    white = np.array([255.0, 255.0, 255.0])
    return np.clip(pix_rgb + scale * white, 0.0, 255.0)


class AnsiColorizer:
    def __init__(
            self,
            colored_fg,
            colored_bg,
            fg_brightness_scale=1.0,
            bg_brightness_scale=0.5,
            use_ansi_256_colors=True):
        self.colored_fg = colored_fg
        self.colored_bg = colored_bg

        self.fg_brightness_scale = fg_brightness_scale
        self.bg_brightness_scale = bg_brightness_scale

        self.use_ansi_256_colors = use_ansi_256_colors
        if use_ansi_256_colors:
            self.set_fg_color_code = lambda r, g, b: set_char_fg_256_color_code(
                rgb_to_ansi_256_id(r, g, b))
            self.set_bg_color_code = lambda r, g, b: set_char_bg_256_color_code(
                rgb_to_ansi_256_id(r, g, b))
        else:
            self.set_fg_color_code = set_char_fg_rgb_color_code
            self.set_bg_color_code = set_char_bg_rgb_color_code

    def create_ansi_prefix(self, pix_rgb):
        ansi_prefix = ""
        if self.colored_fg:
            scaled_rgb_fg = scale_pix_rgb_brightness(
                pix_rgb, self.fg_brightness_scale)
            ansi_prefix += self.set_fg_color_code(
                scaled_rgb_fg[0],
                scaled_rgb_fg[1],
                scaled_rgb_fg[2])
        if self.colored_bg:
            scaled_rgb_bg = scale_pix_rgb_brightness(
                pix_rgb, self.bg_brightness_scale)
            ansi_prefix += self.set_bg_color_code(
                scaled_rgb_bg[0],
                scaled_rgb_bg[1],
                scaled_rgb_bg[2])
        return ansi_prefix
