from typing import Optional

import numpy as np
from PIL import Image, ImageFont
from rendering.ansi_colors_parser import parse_ansi_colors, strip_ansi_codes
from rendering.image import render_symbols_img


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
            colored_fg=True,
            colored_bg=True,
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

    def create_ansi_prefix(self, fg_rgb, bg_rgb):
        ansi_prefix = ""
        if self.colored_bg and bg_rgb is not None:
            scaled_rgb_bg = scale_pix_rgb_brightness(
                bg_rgb, self.bg_brightness_scale)
            ansi_prefix += self.set_bg_color_code(
                scaled_rgb_bg[0],
                scaled_rgb_bg[1],
                scaled_rgb_bg[2])
        if self.colored_fg and fg_rgb is not None:
            scaled_rgb_fg = scale_pix_rgb_brightness(
                fg_rgb, self.fg_brightness_scale)
            ansi_prefix += self.set_fg_color_code(
                scaled_rgb_fg[0],
                scaled_rgb_fg[1],
                scaled_rgb_fg[2])
        return ansi_prefix


def colorize_symbols(
    img_rgb: Image.Image,
    symbol_arr: list[list[str]],
    font: ImageFont.FreeTypeFont,
    colorizer: Optional[AnsiColorizer] = None,
) -> list[list[str]]:
    if not symbol_arr or not symbol_arr[0]:
        return []

    colorizer = colorizer or AnsiColorizer()
    rows, cols = len(symbol_arr), len(symbol_arr[0])

    clean_arr, parsed_arr = [], []
    for row in symbol_arr:
        clean_row, parsed_row = [], []
        for sym in row:
            parsed = parse_ansi_colors(sym)
            clean_row.append(parsed["chars"])
            parsed_row.append(parsed)
        clean_arr.append(clean_row)
        parsed_arr.append(parsed_row)

    base_mask = render_symbols_img(
        clean_arr,
        font=font,
        bg_color=(0, 0, 0),
        fg_color=(255, 255, 255)
    )
    target_size = (
        max(img_rgb.width, base_mask.width),
        max(img_rgb.height, base_mask.height),
    )

    if img_rgb.size != target_size:
        img_rgb = img_rgb.resize(target_size)

    if base_mask.size != target_size:
        base_mask = base_mask.resize(target_size)
    img_w, img_h = img_rgb.size
    patch_w, patch_h = img_w / cols, img_h / rows

    
    base_mask_np = np.array(base_mask.convert("L")).astype(np.float32) / 255.0
    img_np = np.array(img_rgb.convert("RGB")).astype(np.float32)

    colorized_arr = []

    for r in range(rows):
        colorized_row = []
        for c in range(cols):
            clean_sym = clean_arr[r][c]
            parsed_sym = parsed_arr[r][c]

            x0, y0 = int(c * patch_w), int(r * patch_h)
            x1, y1 = int((c + 1) * patch_w), int((r + 1) * patch_h)

            patch = img_np[y0:y1, x0:x1]
            b_mask = base_mask_np[y0:y1, x0:x1]

            orig_fg = parsed_sym.get("fg_color")
            orig_bg = parsed_sym.get("bg_color")

            lum_fg = np.mean(orig_fg) / 255.0 if orig_fg else 1.0
            lum_bg = np.mean(orig_bg) / 255.0 if orig_bg else 1.0

            fg_mask = b_mask * lum_fg
            bg_mask = (1.0 - b_mask) * lum_bg

            fg_rgb = np.zeros(shape=(3,))
            fg_mask_sum = fg_mask.sum()
            if fg_mask_sum:
                fg_rgb = (fg_mask[..., np.newaxis] * patch).sum(axis=(0, 1))
                fg_rgb /= fg_mask_sum

            bg_rgb = np.zeros(shape=(3,))
            bg_mask_sum = bg_mask.sum()
            if bg_mask_sum:
                bg_rgb = (bg_mask[..., np.newaxis] * patch).sum(axis=(0, 1))
                bg_rgb /= bg_mask_sum

            ansi_prefix = colorizer.create_ansi_prefix(fg_rgb, bg_rgb)
            colorized_row.append(f"{ansi_prefix}{clean_sym}{reset_code()}")

        colorized_arr.append(colorized_row)

    return colorized_arr
