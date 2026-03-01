from PIL import Image, ImageDraw
import string
import numpy as np
import random
from modules.ansi_colors_parser import strip_ansi_codes, parse_ansi_colors
from modules.ansi_colorizer import reset_code, set_char_fg_256_color_code, set_char_bg_256_color_code


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


def generate_non_mono_width_aligned_permutations(
        symbols, font, prune_chance=0.0):
    symbol_widths = [(s, font.getbbox(s)[2]) for s in symbols]
    symbol_widths = sorted(symbol_widths, key=lambda s_w: s_w[1])

    max_s, max_width = symbol_widths.pop()
    final_set = [max_s]
    visited = set()
    while len(symbol_widths) > 0:
        s1, w1 = symbol_widths[-1]
        if s1 in visited or random.random() < prune_chance:
            symbol_widths.pop()
            continue
        else:
            visited.add(s1)

        expanded = False
        for s2, w2 in symbol_widths:
            new_s1 = ''
            new_s2 = ''
            new_w = 0
            if w1 + w2 <= max_width:
                new_s1 = s1 + s2
                new_s2 = s2 + s1
                new_w = w1 + w2
            if new_w > 0:
                expanded = True
                symbol_widths.append((new_s1, new_w))
                symbol_widths.append((new_s2, new_w))

        if not expanded:
            final_set.append(s1)
            symbol_widths.pop()

    return final_set
