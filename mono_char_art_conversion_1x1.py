import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale, quantize_rgb, img_rgb_to_max_grayscale
from ansi_colors_parser import *
from ansi_colorizer import AnsiColorizer

def img2char_arr_1x1(img: Image.Image,
                     palette,
                     brightness_palette,
                     dither=DITHER_MODES.NONE,
                     scale_vertically=True,
                     colorize_settings: AnsiColorizer=None) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1] // 2))
    
    if (colorize_settings is None or img.mode != "RGB"):
        img_brightness_mapping_arr = quantize_grayscale(img.convert("L"), len(palette), dither, True, brightness_palette)
        return img_arr2char_arr_1x1(
            img_brightness_mapping_arr, palette, len(palette))


    img_rgb = img
    if colorize_settings.use_ansi_256_colors:
        img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb, dtype=np.float32)
    img_brightness_mapping_arr = quantize_grayscale(img_rgb_to_max_grayscale(img_rgb), len(palette), dither, True, brightness_palette)

    return img_arr2char_arr_1x1(img_brightness_mapping_arr, palette, len(
        palette), img_rgb_arr, colorize_settings)


def img_arr2char_arr_1x1(img_arr: np.ndarray,
                         palette: list[str],
                         img_shades=256,
                         img_rgb_arr=None,
                         colorize_settings: AnsiColorizer=None) -> list[list[str]]:
    palette_interval = img_shades / len(palette)
    char_arr = []
    for y in range(img_arr.shape[0]):
        char_arr.append([])
        for x in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[y][x] // palette_interval)]
            if (len(palette_cell) > 1):
                char_arr[-1].append(palette_cell[random.randrange(len(palette_cell))])
            else:
                char_arr[-1].append(palette_cell[0])
            if img_rgb_arr is not None and colorize_settings is not None:
                pix_rgb = img_rgb_arr[y][x]
                char_arr[-1][-1] = colorize_settings.create_ansi_prefix(pix_rgb) + char_arr[-1][-1]
    return char_arr
