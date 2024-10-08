import numpy as np
from PIL import Image
import random
from img_processing import *
from mono_char_art_conversion_wxh import scale_kernel
from ansi_colors_parser import *
from ansi_colorizer import AnsiColorizer


def quantize_grayscale_1x2(img: Image.Image, img_colors: tuple[int, int],
                           dither=DITHER_MODES.NONE, return_palette_map=False,
                           palette: np.ndarray = None) -> list[list[int, int]]:
    if (img.mode != "L"):
        raise Exception("img mode should be \"L\"")
    if (img_colors[0] <= 0 or img_colors[1] <= 0):
        raise Exception("img_colors should be > 0")
    if (palette is not None and (img_colors[0] != palette.shape[0]
                                 or img_colors[1] != palette.shape[1] or palette.shape[2] < 2)):
        raise Exception(
            "palette should be of shape img_colors with two subfields per cell")

    if (palette is None):
        linspace_y = np.linspace(0, 1, img_colors[0])
        linspace_x = np.linspace(0, 1, img_colors[1])
        palette = np.full((img_colors[0], img_colors[1], 2), 0.0)
        for y in range(0, img_colors[0]):
            for x in range(0, img_colors[1]):
                palette[y][x][0] = linspace_y[y]
                palette[y][x][1] = linspace_x[x]

    if (dither == DITHER_MODES.JJN):
        scaled_jjn_k, jjn_k_offset = scale_kernel(jjn_k, 1, 2, 2)
    if (dither == DITHER_MODES.FS):
        scaled_fs_k, fs_k_offset = scale_kernel(fs_k, 1, 2, 1)

    img_arr = np.array(img) / 255
    color_step_0 = 1 / img_colors[0]
    color_step_1 = 1 / img_colors[1]
    palette_map = np.zeros((img_arr.shape[0]//2, img_arr.shape[1], 2),
                           dtype=np.int32)

    for y in range(1, img_arr.shape[0], 2):
        for x in range(0, img_arr.shape[1]):
            c0 = img_arr[y - 1][x]
            c1 = img_arr[y][x]

            c0_new = c0
            c1_new = c1
            if (dither == DITHER_MODES.BAYER):
                c0_new = apply_threshold_map(
                    c0, dither_bayer_m, color_step_0, x, y - 1)
                c1_new = apply_threshold_map(
                    c1, dither_bayer_m, color_step_1, x, y)

            c0_new_idx = int(c0_new / color_step_0)
            c0_new_idx = min(len(palette) - 1, max(0, c0_new_idx))

            c1_new_idx = int(c1_new / color_step_1)
            c1_new_idx = min(len(palette[c0_new_idx]) - 1, max(0, c1_new_idx))
            palette_map[y//2][x][0] = c0_new_idx
            palette_map[y//2][x][1] = c1_new_idx

            c0_new = palette[c0_new_idx][c1_new_idx][0]
            c1_new = palette[c0_new_idx][c1_new_idx][1]

            if (dither == DITHER_MODES.JJN):
                apply_error_diff_window(np.array(
                    [c0_new, c1_new]), 1, 2, img_arr, x, y - 1, scaled_jjn_k, jjn_k_offset)

            if (dither == DITHER_MODES.FS):
                apply_error_diff_window(
                    np.array([c0_new, c1_new]), 1, 2, img_arr, x, y - 1, scaled_fs_k, fs_k_offset)

            img_arr[y - 1][x] = c0_new
            img_arr[y][x] = c1_new

    if (return_palette_map):
        return palette_map

    img_arr = np.array(img_arr * 255, dtype=np.ubyte)
    return Image.frombytes("L", (img_arr.shape[1], img_arr.shape[0]), img_arr)


def img2char_arr_1x2(img: Image.Image,
                     palette: list[list[str]],
                     brightness_palette,
                     dither=DITHER_MODES.NONE,
                     colorize_settings: AnsiColorizer = None) -> list[list[str]]:

    if (colorize_settings is None or img.mode != "RGB"):
        img_arr = quantize_grayscale_1x2(
            img.convert("L"), (len(palette), len(
                palette[0])), dither, True, np.array(brightness_palette))
        return img_arr2char_arr_1x2(
            img_arr, palette, (len(palette), len(palette[0])))

    img_rgb = img
    if colorize_settings.use_ansi_256_colors:
        img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb, dtype=np.float32)
    img_arr = quantize_grayscale_1x2(img_rgb_to_max_grayscale(img_rgb), (len(
        palette), len(palette[0])), dither, True, np.array(brightness_palette))
    return img_arr2char_arr_1x2(
        img_arr, palette, (len(palette), len(
            palette[0])), img_rgb_arr, colorize_settings)


def img_arr2char_arr_1x2(img_arr: np.ndarray, palette: list[list[str]], img_colors=(
        256, 256), img_rgb_arr=None, colorize_settings: AnsiColorizer = None) -> list[list[str]]:
    palette_y_interval = img_colors[0] // len(palette)
    palette_x_interval = img_colors[1] // len(palette[0])
    char_arr = []
    for y in range(img_arr.shape[0]):
        char_arr.append([])
        for x in range(img_arr.shape[1]):
            palette_cell = palette[img_arr[y][x][0] // palette_y_interval][img_arr[y][x][1] // palette_x_interval]
            if (len(palette_cell) > 1):
                char_arr[-1].append(palette_cell[random.randint(0,
                                    len(palette_cell) - 1)])
            else:
                char_arr[-1].append(palette_cell[0])
            if img_rgb_arr is not None and colorize_settings is not None:
                pix_rgb = (img_rgb_arr[y*2][x] + img_rgb_arr[y*2 + 1][x]) / 2
                char_arr[-1][-1] = colorize_settings.create_ansi_prefix(
                    pix_rgb) + char_arr[-1][-1]

    return char_arr
