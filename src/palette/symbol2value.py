import sys
import string
import bisect
import random

import numpy as np
from PIL import Image, ImageDraw

from rendering.ansi_colors_parser import strip_ansi_codes, parse_ansi_colors
from rendering.ansi_colorizer import reset_code, set_char_fg_256_color_code, set_char_bg_256_color_code


def get_asciis():
    return list(filter(lambda a: a.isprintable(), string.printable))


def add_ansi_256_colors_inversions(
        symbols: list[str],
        fg_ansi_256_id=7,
        bg_ansi_256_id=0):
    char_set_with_inversions = []
    for a in symbols:
        char_set_with_inversions.append(
            set_char_fg_256_color_code(fg_ansi_256_id)
            + set_char_bg_256_color_code(bg_ansi_256_id)
            + a + reset_code())
        char_set_with_inversions.append(
            set_char_fg_256_color_code(bg_ansi_256_id)
            + set_char_bg_256_color_code(fg_ansi_256_id)
            + a + reset_code())
    return char_set_with_inversions


def add_ansi_256_colors_grayscale(
        symbols: list[str],
        num_colors=24,
        use_fg=True,
        use_bg=True):
    if (num_colors > 24 or num_colors < 2):
        raise Exception("num_colors should be < 24 and > 1")
    if (not (use_fg or use_bg)):
        return symbols

    ids = np.unique(
        np.linspace(
            232,
            255,
            num_colors,
            endpoint=True,
            dtype=np.int32))

    fgs = [(-1, '')]
    if use_fg:
        fgs = map(set_char_fg_256_color_code, ids)
        fgs = list(zip(ids, fgs))
    bgs = [(-1, '')]
    if use_bg:
        bgs = map(set_char_bg_256_color_code, ids)
        bgs = list(zip(ids, bgs))

    prefixes = []
    for fg_id, fg in fgs:
        for bg_id, bg in bgs:
            if (fg_id != bg_id):
                prefixes.append(fg + bg)

    grayscale_syms = []
    for sym in symbols:
        for prefix in prefixes:
            grayscale_syms.append(prefix + sym + reset_code())

    return grayscale_syms


def add_ansi_256_colors(symbols):
    colored_syms = []
    for sym in symbols:
        for fg in range(0, 256):
            for bg in range(0, 256):
                if (fg == bg):
                    continue
                colored = set_char_fg_256_color_code(fg) \
                    + set_char_bg_256_color_code(bg) \
                    + sym \
                    + reset_code()
                colored_syms.append(colored)
    return colored_syms


def normalize_values(colors: list[np.ndarray]):
    max_vals = np.max(np.array(colors), axis=0)
    max_vals = np.clip(max_vals, 0.001, 1.0)
    return [b / max_vals for b in colors]


def make_symbol2value_map(
        symbols,
        font,
        val_width,
        val_height,
        bg_color=(0, 0, 0),
        fg_color=(255, 255, 255),
        grayscale=True,
        normalize=False) -> dict[str, np.ndarray]:

    width, height = 0, 0
    for sym in symbols:
        sym = strip_ansi_codes(sym)
        width = max(width, font.getbbox(sym)[2])
        height = max(height, font.getbbox(sym)[3])

    colors = []
    for sym in symbols:
        sym_info = parse_ansi_colors(sym)
        if (sym_info.get("fg_color") is None):
            sym_info["fg_color"] = fg_color
        if (sym_info.get("bg_color") is None):
            sym_info["bg_color"] = bg_color

        img = Image.new(
            mode="RGB",
            size=(width, height),
            color=sym_info["bg_color"])

        img_d = ImageDraw.Draw(img)
        img_d.text(
            (width / 2, height / 2),
            sym_info["chars"],
            font=font,
            fill=sym_info["fg_color"],
            anchor='mm')

        if grayscale:
            img = img.convert("L")
        res_img = img.resize((val_width, val_height), Image.Resampling.BOX)
        res_arr = np.array(res_img) / 255
        colors.append(res_arr)

    if normalize:
        colors = normalize_values(colors)

    return {s: c for s, c in zip(symbols, colors)}


def stream_width_aligned_permutations(symbols, font, skip_chance=0.0):
    str_widths = [(s, font.getbbox(s)[2]) for s in symbols]
    str_widths = sorted(str_widths, key=lambda s_w: s_w[1])

    max_s, max_width = str_widths.pop()
    yield max_s

    widths = [w for _, w in str_widths]
    symbs = [s for s, _ in str_widths]

    yielded = set()

    while str_widths:
        curr_str, curr_width = str_widths.pop()
        if curr_str in yielded:
            continue

        rest_width = max_width - curr_width

        max_fit_idx = bisect.bisect_right(widths, rest_width)

        if max_fit_idx == 0 and random.random() > skip_chance:
            yield curr_str
            continue

        for i in range(max_fit_idx):
            next_str = curr_str + symbs[i]
            next_width = curr_width + widths[i]
            str_widths.append((next_str, next_width))


def symbols_sorted(symbols, font):
    symbol2brightness = make_symbol2value_map(
        symbols=symbols,
        font=font,
        val_height=1,
        val_width=1,
        normalize=True)
    symb_brs = [(s, b[0][0]) for s, b in symbol2brightness.items()]
    symb_brs = sorted(symb_brs, key=lambda s_b: s_b[1])
    return list(map(lambda s_b: s_b[0], symb_brs))

