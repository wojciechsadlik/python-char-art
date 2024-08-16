from PIL import Image, ImageDraw, ImageOps, ImageEnhance, ImageFilter, ImageChops
import string
import numpy as np
import random
from ansi_colors_parser import *
from ansi_colorizer import set_char_fg_256_color_code, set_char_bg_256_color_code


def get_asciis():
    return list(filter(lambda a: a.isprintable(), string.printable))


def get_char_set_with_inversions(char_set, fg_ansi_256_id=7, bg_ansi_256_id=0):
    char_set_with_inversions = []
    for a in char_set:
        char_set_with_inversions.append(set_char_fg_256_color_code(
            fg_ansi_256_id) + set_char_bg_256_color_code(bg_ansi_256_id) + a)
        char_set_with_inversions.append(set_char_fg_256_color_code(
            bg_ansi_256_id) + set_char_bg_256_color_code(fg_ansi_256_id) + a)
    return char_set_with_inversions


def get_ansi_256_colors_grayscale_set(char_set, num_colors=24):
    if (num_colors > 24 or num_colors < 2):
        raise Exception("num_colors should be < 24 and > 1")
    ids = np.unique(
        np.linspace(
            232,
            255,
            num_colors,
            endpoint=True,
            dtype=np.int32))
    grayscale_set = []
    for a in char_set:
        for fg in ids:
            for bg in ids:
                if (fg != bg):
                    grayscale_set.append(set_char_fg_256_color_code(
                        fg) + set_char_bg_256_color_code(bg) + a)
    return grayscale_set


def get_ansi_256_colors_set(char_set):
    colored_set = []
    for a in char_set:
        for fg in range(0, 256):
            for bg in range(0, 256):
                if (fg != bg):
                    colored_set.append(set_char_fg_256_color_code(
                        fg) + set_char_bg_256_color_code(bg) + a)
    return colored_set


def max_brighntess_val(brightnesses):
    max_b_glob = 0
    for b in brightnesses:
        max_b = max(b.flatten())
        if max_b_glob < max_b:
            max_b_glob = max_b
    return max_b_glob


def max_brightness_per_pos(brightnesses):
    max_br = np.zeros_like(brightnesses[0])
    for b in brightnesses:
        for y in range(b.shape[0]):
            for x in range(b.shape[1]):
                if b[y][x] > max_br[y][x]:
                    max_br[y][x] = b[y][x]
    return max_br


def normalize_brightness_map(brightnesses):
    return [b / max_brighntess_val(brightnesses) for b in brightnesses]


def generate_brightness_map(
    char_set, font, window_wh_size, bg_color=(0, 0, 0),
    char_color=(255, 255, 255), grayscale=True, normalize=False):
    
    width, height = 0, 0
    for char in char_set:
        char = strip_ansi_codes(char)
        width = max(width, font.getbbox(char)[2])
        height = max(height, font.getbbox(char)[3])

    width -= width % window_wh_size[0]
    height -= height % window_wh_size[1]

    brightnesses = []
    for char in char_set:
        char_info = parse_ansi_colors(char)
        if (char_info.get("fg_color") is None):
            char_info["fg_color"] = char_color
        if (char_info.get("bg_color") is None):
            char_info["bg_color"] = bg_color

        img = Image.new(
            mode="RGB",
            size=(
                width,
                height),
            color=char_info["bg_color"])
        img_d = ImageDraw.Draw(img)
        img_d.text(
            (width / 2,
             height / 2),
            char_info["char"],
            font=font,
            fill=char_info["fg_color"],
            anchor='mm')
        if (grayscale):
            img = img.convert("L")
            res_img = img.resize(window_wh_size, Image.Resampling.BICUBIC)
            res_arr = np.array(res_img) / 255
            
            if (window_wh_size[1] > 2):
                for i in range(res_arr.shape[1]):
                    res_arr[0][i] = max(res_arr[0][i],
                                        (res_arr[0][i] + res_arr[1][i]) / 2)
                    res_arr[-1][i] = max(res_arr[-1][i],
                                        (res_arr[-1][i] + res_arr[-2][i]) / 2)
        else:
            res_img = img.resize(window_wh_size, Image.Resampling.BICUBIC)
            res_arr = np.array(res_img) / 255

        brightnesses.append(res_arr)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(char_set, brightnesses)}


def generate_non_mono_brightness_map(
        char_set,
        font,
        max_width,
        height,
        bg_color=0,
        char_color=255,
        normalize=False):
    max_width_in_set = max([font.getbbox(c)[2] for c in char_set])
    width_scale = max_width / max_width_in_set
    brightnesses = []
    for char in char_set:
        _, _, char_width, char_height = font.getbbox(char)
        img = Image.new(
            mode="L",
            size=(
                char_width,
                char_height),
            color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((char_width / 2, char_height / 2), char,
                   font=font, fill=char_color, anchor='mm')
        res_img = img.resize(
            (int(
                char_width *
                width_scale),
                height),
            Image.Resampling.BICUBIC)
        brightnesses.append(np.array(res_img) / 255)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(char_set, brightnesses)}


def generate_non_mono_width_aligned_permutations(char_set, font, prune=0.0):
    width_sorted_set = sorted([(c, font.getbbox(c)[2])
                              for c in char_set], key=lambda c_w: c_w[1])
    max_s, max_width = width_sorted_set.pop()
    final_set = [max_s]
    visited = set()
    while len(width_sorted_set) > 0:
        s1, w1 = width_sorted_set[-1]
        if s1 in visited or random.random() < prune:
            width_sorted_set.pop()
            continue
        else:
            visited.add(s1)

        expanded = False
        for s2, w2 in width_sorted_set:
            new_s1 = ''
            new_s2 = ''
            new_w = 0
            if w1 + w2 <= max_width:
                new_s1 = s1 + s2
                new_s2 = s2 + s1
                new_w = w1 + w2
            if new_w > 0:
                expanded = True
                width_sorted_set.append((new_s1, new_w))
                width_sorted_set.append((new_s2, new_w))

        if not expanded:
            final_set.append(s1)
            width_sorted_set.pop()

    return final_set


def generate_non_mono_multi_char_brightness_map(
        char_set,
        font,
        width,
        height,
        bg_color=0,
        char_color=255,
        normalize=False,
        prune=0.0):
    width_aligned_set = generate_non_mono_width_aligned_permutations(
        char_set, font, prune)

    brightnesses = []
    for char_str in width_aligned_set:
        _, _, char_str_width, char_str_height = font.getbbox(char_str)
        img = Image.new(
            mode="L",
            size=(
                char_str_width,
                char_str_height),
            color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((char_str_width / 2, char_str_height / 2),
                   char_str, font=font, fill=char_color, anchor='mm')
        res_img = img.resize((width, height), Image.Resampling.BICUBIC)
        brightnesses.append(np.array(res_img) / 255)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(width_aligned_set, brightnesses)}


def generate_1_1_palette(
        char_set,
        font,
        bins=12,
        bg_color=0,
        char_color=255,
        normalize=False):
    char_to_brightness = generate_brightness_map(
        char_set, font, (1, 1), bg_color, char_color, True, normalize)
    char_to_brightness = dict(
        map(lambda c_b: (c_b[0], c_b[1][0][0]), char_to_brightness.items()))
    sorted_map = sorted(char_to_brightness.items(), key=lambda c_b: c_b[1])

    br_max = max_brighntess_val(char_to_brightness.values())
    char_bins = [[] for _ in range(bins)]
    br_step = br_max / (bins - 1)
    for char, char_br in sorted_map:
        bin_index = int(round(char_br / br_step))
        char_bins[bin_index].append(char)

    if (len(char_bins[0]) == 0):
        char_bins[0].append(sorted_map[0][0])

    for i in range(1, bins):
        if (len(char_bins[i]) == 0):
            char_bins[i].append(char_bins[i - 1][-1])

    bin_to_brightness = []
    for bin in char_bins:
        bin_to_brightness.append(np.mean([char_to_brightness[c] for c in bin]))

    return char_bins, bin_to_brightness


def generate_1_2_palette(
        char_palette,
        font,
        bins=(
            9,
            9),
    bg_color=0,
    char_color=255,
        normalize=False):
    char_to_brightness = generate_brightness_map(
        char_palette, font, (1, 2), bg_color, char_color, True, normalize)

    x_bins = bins[0]
    y_bins = bins[1]

    char_bins = [[[] for _ in range(x_bins)] for _ in range(y_bins)]
    br_max = max_brightness_per_pos(list(char_to_brightness.values()))
    br_y_step = br_max[0][0] / (y_bins - 1)
    br_x_step = br_max[1][0] / (x_bins - 1)
    for char, char_br in char_to_brightness.items():
        bin_y_index = int(round(char_br[0][0] / br_y_step))
        bin_x_index = int(round(char_br[1][0] / br_x_step))
        char_bins[bin_y_index][bin_x_index].append(char)

    def splice_cell(top, left):
        if len(top) > 0:
            res = top[0]
        else:
            res = left[0]
        for a in top:
            if (np.mean(char_to_brightness[res])
                    < np.mean(char_to_brightness[a])):
                res = a

        for a in left:
            if (np.mean(char_to_brightness[res])
                    < np.mean(char_to_brightness[a])):
                res = a
        return res

    for y in range(y_bins):
        for x in range(x_bins):
            if len(char_bins[y][x]) == 0:
                if x > 0 and y > 0:
                    char_bins[y][x].append(splice_cell(
                        char_bins[y - 1][x], char_bins[y][x - 1]))
                elif x > 0:
                    char_bins[y][x].append(
                        splice_cell([], char_bins[y][x - 1]))
                elif y > 0:
                    char_bins[y][x].append(
                        splice_cell(char_bins[y - 1][x], []))

    bin_to_brightness = []
    for row in char_bins:
        bin_to_brightness.append([])
        for bin in row:
            bin_to_brightness[-1].append(np.mean([char_to_brightness[c]
                                         for c in bin], axis=0))

    return char_bins, bin_to_brightness
