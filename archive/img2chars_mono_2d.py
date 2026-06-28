import numpy as np
from PIL import Image
import random
from image.processing import DITHER_MODES, quantize_grayscale, \
    quantize_rgb, img_rgb_to_max_grayscale
from rendering.ansi_colorizer import AnsiColorizer, reset_code


def img2char_arr_2d(img: Image.Image,
                    bin_palette: list[list[list[str]]],
                    dither=DITHER_MODES.NONE,
                    ansi_colorizer: AnsiColorizer = None) -> list[list[str]]:

    if (ansi_colorizer is None or img.mode != "RGB"):
        palette_ids = img2palette_ids(
            img=img,
            bin_palette=bin_palette,
            dither=dither
        )
        return palette_ids2char_arr(
            palette_ids_arr=palette_ids,
            bin_palette=bin_palette)

    img_rgb = img
    if ansi_colorizer.use_ansi_256_colors:
        img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb, dtype=np.float32)
    palette_ids = img2palette_ids(
        img=img_rgb,
        bin_palette=bin_palette,
        dither=dither
    )
    return palette_ids2char_arr(
        palette_ids_arr=palette_ids,
        bin_palette=bin_palette,
        img_rgb_arr=img_rgb_arr,
        ansi_colorizer=ansi_colorizer)


def img2palette_ids(img: Image.Image,
                    bin_palette: list[list[list[str]]],
                    dither=DITHER_MODES.NONE) -> list[list[tuple]]:
    img_colors = max(len(bin_palette), len(bin_palette[0]))
    if (img.mode != "RGB"):
        img_arr = quantize_grayscale(
            img=img.convert("L"),
            img_colors=img_colors,
            dither=dither,
            return_palette_map=True)
    else:
        img_arr = quantize_grayscale(
            img=img_rgb_to_max_grayscale(img),
            img_colors=img_colors,
            dither=dither,
            return_palette_map=True)

    y_palette_interval = img_colors / len(bin_palette)
    x_palette_interval = img_colors / len(bin_palette[0])
    palette_ids_arr = []
    for y in range(1, img_arr.shape[0], 2):
        palette_ids_arr.append([])
        for x in range(img_arr.shape[1]):
            top_pix = img_arr[y - 1][x]
            btm_pix = img_arr[y][x]
            y_palette_idx = int(top_pix / y_palette_interval)
            x_palette_idx = int(btm_pix / x_palette_interval)

            palette_ids_arr[-1].append((y_palette_idx, x_palette_idx))

    return palette_ids_arr


def img_arr2palette_ids(img_arr: np.ndarray,
                        palette: list[list[list[str]]],
                        img_colors_top_btm=(256, 256)) -> list[list[tuple]]:
    y_palette_interval = img_colors_top_btm[0] / len(palette)
    x_palette_interval = img_colors_top_btm[1] / len(palette[0])
    palette_ids_arr = []
    for y in range(1, img_arr.shape[0], 2):
        palette_ids_arr.append([])
        for x in range(img_arr.shape[1]):
            top_pix = img_arr[y - 1][x]
            btm_pix = img_arr[y][x]
            y_palette_idx = int(top_pix / y_palette_interval)
            x_palette_idx = int(btm_pix / x_palette_interval)

            palette_ids_arr[-1].append((y_palette_idx, x_palette_idx))

    return palette_ids_arr


def palette_ids2char_arr(
        palette_ids_arr: list[list[tuple]],
        bin_palette: list[list[list[str]]],
        img_rgb_arr=None,
        ansi_colorizer: AnsiColorizer = None) -> list[list[str]]:
    char_arr = []
    for y, row in enumerate(palette_ids_arr):
        char_arr.append([])
        for x, palette_idx in enumerate(row):
            palette_cell = bin_palette[palette_idx[0]][palette_idx[1]]

            char = palette_cell[0]
            if (len(palette_cell) > 1):
                char = palette_cell[random.randrange(len(palette_cell))]
            char_arr[-1].append(char)

            if img_rgb_arr is not None and ansi_colorizer is not None:
                pix_rgb = (img_rgb_arr[y * 2][x] +
                           img_rgb_arr[y * 2 + 1][x]) / 2
                char_arr[-1][-1] = ansi_colorizer.create_ansi_prefix(
                    pix_rgb) + char_arr[-1][-1]
        if ansi_colorizer:
            char_arr[-1].append(reset_code())

    return char_arr
