import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale, quantize_rgb, img_rgb_to_max_grayscale
from ansi_colors_parser import *
from ansi_colorizer import AnsiColorizer


def find_match(win, char_to_brightness_map, randomize=False):
    max_dist = np.linalg.norm(np.ones(win.shape))
    max_sim = 0
    max_char = ''
    max_br = []
    close_char = []
    close_br = []
    #print(char_to_brightness_map.keys())
    for char, br in char_to_brightness_map.items():
        sim = 1 - np.linalg.norm(win - br) / max_dist

        if (sim > max_sim):
            max_sim = sim
            max_char = char
            max_br = br
        if (sim > 0.5):
            close_char.append(char)
            close_br.append(br)

    if (len(close_char) > 1 and randomize):
        rand_char_id = random.randrange(len(close_char))
        return close_char[rand_char_id], close_br[rand_char_id]
    return max_char, max_br


def img2char_arr_1x1(img: Image.Image,
                     palette,
                     brightness_palette,
                     dither=DITHER_MODES.NONE,
                     scale_vertically=True,
                     colorize_settings: AnsiColorizer = None) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1] // 2))

    if (colorize_settings is None or img.mode != "RGB"):
        img_brightness_mapping_arr = quantize_grayscale(
            img.convert("L"), len(palette), dither, True, brightness_palette)
        return img_palette_arr2char_arr_1x1(
            img_brightness_mapping_arr, palette, len(palette))

    img_rgb = img
    if colorize_settings.use_ansi_256_colors:
        img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb, dtype=np.float32)
    img_brightness_mapping_arr = quantize_grayscale(img_rgb_to_max_grayscale(
        img_rgb), len(palette), dither, True, brightness_palette)

    return img_palette_arr2char_arr_1x1(img_brightness_mapping_arr, palette, len(
        palette), img_rgb_arr, colorize_settings)


def img_palette_arr2char_arr_1x1(img_arr: np.ndarray,
                         palette: list[str],
                         img_shades=256,
                         img_rgb_arr=None,
                         colorize_settings: AnsiColorizer = None,
                         detailed_img_arr=None,
                         detailed_mapping=None,
                         detailed_mapping_res=None) -> list[list[str]]:
    use_detailed_mapping = (detailed_img_arr is not None
                            and detailed_mapping is not None
                            and detailed_mapping_res is not None)
    palette_interval = img_shades // len(palette)
    char_arr = []
    for y in range(img_arr.shape[0]):
        char_arr.append([])
        for x in range(img_arr.shape[1]):
            palette_cell = palette[img_arr[y][x] // palette_interval]
            if (len(palette_cell) > 1):
                if use_detailed_mapping:
                    win = detailed_img_arr[y:y+detailed_mapping_res[1],
                                            x:x+detailed_mapping_res[0]]
                    char, _ = find_match(win, {c[0]: c[1] for c in detailed_mapping.items() if c[0] in palette_cell})
                    char_arr[-1].append(char)
                else:
                    char_arr[-1].append(palette_cell[random.randrange(len(palette_cell))])
            else:
                char_arr[-1].append(palette_cell[0])
            if img_rgb_arr is not None and colorize_settings is not None:
                pix_rgb = img_rgb_arr[y][x]
                char_arr[-1][-1] = colorize_settings.create_ansi_prefix(
                    pix_rgb) + char_arr[-1][-1]
    return char_arr
