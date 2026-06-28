import numpy as np
from PIL import Image
import random
from image.processing import DITHER_MODES, quantize_grayscale, \
    quantize_rgb, img_rgb_to_max_grayscale
from rendering.ansi_colorizer import AnsiColorizer, reset_code


def img2char_arr_1d(img: Image.Image,
                    bin_palette: list[list[str]],
                    bin_vals: list[float],
                    dither=DITHER_MODES.NONE,
                    scale_vertically=True,
                    ansi_colorizer: AnsiColorizer = None) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1] // 2))

    if (ansi_colorizer is None or img.mode != "RGB"):
        img_palette_map = quantize_grayscale(
            img=img.convert("L"),
            img_colors=len(bin_palette),
            dither=dither,
            return_palette_map=True,
            palette=bin_vals)
        return img_arr2char_arr(
            img_arr=img_palette_map,
            palette=bin_palette,
            img_colors=len(bin_palette))

    img_rgb = img
    if ansi_colorizer.use_ansi_256_colors:
        img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb, dtype=np.float32)
    img_palette_map = quantize_grayscale(
        img=img_rgb_to_max_grayscale(img_rgb),
        img_colors=len(bin_palette),
        dither=dither,
        return_palette_map=True,
        palette=bin_vals)

    return img_arr2char_arr(
        img_arr=img_palette_map,
        palette=bin_palette,
        img_colors=len(bin_palette),
        img_rgb_arr=img_rgb_arr,
        ansi_colorizer=ansi_colorizer)


def img_arr2char_arr(img_arr: np.ndarray,
                     palette: list[list[str]],
                     img_colors=256,
                     img_rgb_arr=None,
                     ansi_colorizer: AnsiColorizer = None) -> list[list[str]]:
    palette_interval = img_colors / len(palette)
    char_arr = []
    for y in range(img_arr.shape[0]):
        char_arr.append([])
        for x in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[y][x] / palette_interval)]
            char = palette_cell[0]
            if (len(palette_cell) > 1):
                char = palette_cell[random.randrange(len(palette_cell))]
            char_arr[-1].append(char)
            if img_rgb_arr is not None and ansi_colorizer is not None:
                pix_rgb = img_rgb_arr[y][x]
                char_arr[-1][-1] = ansi_colorizer.create_ansi_prefix(
                    pix_rgb) + char_arr[-1][-1]
        if ansi_colorizer:
            char_arr[-1].append(reset_code())
    return char_arr
